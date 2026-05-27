import os 
import ctypes
import platform

import torch
import numpy as np


_FALSE_ENV_VALUES = {"0", "false", "no", "off"}


def _read_env_bool(names, default: bool) -> bool:
    for name in names:
        raw_value = os.environ.get(str(name))
        if raw_value is None:
            continue
        return raw_value.strip().lower() not in _FALSE_ENV_VALUES
    return bool(default)


def _read_env_int(name: str) -> int | None:
    raw_value = os.environ.get(str(name))
    if raw_value is None:
        return None
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return None


class LattigoFunction:
    """Helper to wrap ctypes functions with argument and return types."""
    def __init__(self, func, argtypes, restype):
        self.func = func
        self.func.argtypes = argtypes 
        self.func.restype = restype

    def __call__(self, *args):
        c_args = []
        for arg in args:
            curr_argtype = self.func.argtypes[len(c_args)]
            c_arg = self.convert_to_ctypes(arg, curr_argtype)
            if isinstance(c_arg, tuple):
                c_args.extend(c_arg)
            else:
                c_args.append(c_arg)
                
        c_result = self.func(*c_args)
        py_result = self.convert_from_ctypes(c_result)
        
        # If the result is a list, then we'll need to manually free the
        # memory we allocated for this list in Go with the below. We'll
        # defer freeing byte data (from serialization) until after that
        # data has been saved to HDF5.
        if isinstance(py_result, list):
            LattigoFunction.FreeCArray(
                ctypes.cast(c_result.Data, ctypes.c_void_p))

        return py_result

    @torch._dynamo.disable
    def convert_to_ctypes(self, arg, typ):
        if isinstance(arg, int) and typ == ctypes.c_int:
            return ctypes.c_int(arg)
        elif isinstance(arg, int) and typ == ctypes.c_ulong:
            return ctypes.c_ulong(arg)
        elif isinstance(arg, int) and typ == ctypes.c_ulonglong:
            return ctypes.c_ulonglong(arg)
        elif isinstance(arg, float):
            return ctypes.c_float(arg)
        elif isinstance(arg, str):
            return arg.encode('utf-8')
        elif (isinstance(arg, np.ndarray) and 
            arg.dtype == np.uint8 and 
            typ == ctypes.POINTER(ctypes.c_ubyte)):
            ptr = arg.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
            return (ptr, len(arg))
        elif isinstance(arg, list):
            if typ == ctypes.POINTER(ctypes.c_int):
                return ((ctypes.c_int * len(arg))(*arg), len(arg))
            elif typ == ctypes.POINTER(ctypes.c_float):
                return ((ctypes.c_float * len(arg))(*arg), len(arg))
            elif typ == ctypes.POINTER(ctypes.c_ulong):
                return ((ctypes.c_ulong * len(arg))(*arg), len(arg))
            elif typ == ctypes.POINTER(ctypes.c_ulonglong):
                return ((ctypes.c_ulonglong * len(arg))(*arg), len(arg))
            elif typ == ctypes.POINTER(ctypes.c_ubyte):
                return ((ctypes.c_ubyte * len(arg))(*arg), len(arg))
            else:
                raise ValueError("Unexpected list type to convert.")
        else:
            return arg
            
    def convert_from_ctypes(self, res):
        if type(res) == ctypes.c_int:
            return int(res)
        elif type(res) == ctypes.c_float:
            return float(res)
        elif type(res) == ArrayResultFloat:
            length = int(res.Length)
            return [float(res.Data[i]) for i in range(length)]
        elif type(res) in (ArrayResultInt, ArrayResultUInt64):
            length = int(res.Length)
            return [int(res.Data[i]) for i in range(length)]
        elif type(res) == ArrayResultDouble:
            length = int(res.Length)
            return [float(res.Data[i]) for i in range(length)]
        elif type(res) == ArrayResultByte:
            length = int(res.Length)
            # Create numpy array directly from the C buffer
            buffer = ctypes.cast(
                res.Data, 
                ctypes.POINTER(ctypes.c_ubyte * length)
            ).contents
            array = np.frombuffer(buffer, dtype=np.uint8)
            return array, res.Data
        else:
            return res


class LattigoLibrary:
    """A class to manage loading and interfacing with Lattigo."""
    def __init__(self):
        self.lib = self._load_library()
        self.load_plaintext_diagonals_requires_payload = True
        self.saved_io_prefetch_requires_device_memory = False
        self.saved_io_host_predecode_enabled = _read_env_bool(
            (
                "ORION_SAVED_IO_HOST_PREDECODE",
                "ORION_LATTIGO_SAVED_IO_HOST_PREDECODE",
            ),
            True,
        )
        self.saved_io_host_predecode_supported = False
        self.supports_index_only_linear_transform_load = True
        self.memory_bounded_unified_transforms = _read_env_bool(
            (
                "ORION_LATTIGO_MEMORY_BOUNDED_UNIFIED_TRANSFORMS",
                "ORION_LATTIGO_MEMORY_BOUNDED_COMPILE",
            ),
            True,
        )
        self.memory_bounded_unified_evaluate = _read_env_bool(
            (
                "ORION_LATTIGO_MEMORY_BOUNDED_UNIFIED_EVALUATE",
                "ORION_LATTIGO_MEMORY_BOUNDED_EVAL",
            ),
            True,
        )
        eval_budget = _read_env_int("ORION_LATTIGO_UNIFIED_EVAL_BUDGET_BYTES")
        if eval_budget is not None:
            self.unified_transform_eval_budget_bytes = max(1, int(eval_budget))

    def _load_library(self):
        try:
            # Determine library name based on platform
            if platform.system() == "Linux":
                lib_name = "lattigo-linux.so"
            elif platform.system() == "Darwin":  # macOS
                if platform.machine().lower() in ("arm64", "aarch64"):
                    lib_name = "lattigo-mac-arm64.dylib"
                else:
                    lib_name = "lattigo-mac.dylib"
            elif platform.system() == "Windows":
                lib_name = "lattigo-windows.dll"
            else:
                raise RuntimeError("Unsupported platform")
                        
            # Standard path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            lib_path = os.path.join(current_dir, lib_name)            
            return ctypes.CDLL(lib_path)
            
        except OSError as e:
            raise RuntimeError(f"Failed to load Lattigo library: {e}")

    def _find_library(self, root_dir, lib_name):
        """Recursively search for the library file"""
        for root, _, files in os.walk(root_dir):
            if lib_name in files:
                return os.path.join(root, lib_name)
        raise FileNotFoundError(f"Library {lib_name} not found in {root_dir}")
                    
    def setup_bindings(self, orion_params):
        """
        Declares the functions from the Lattigo shared library and sets their
        argument and return types.
        """
        self.setup_scheme(orion_params)
        self.setup_tensor_binds()
        self.setup_key_generator()
        self.setup_encoder()
        self.setup_encryptor()
        self.setup_evaluator()
        self.setup_poly_evaluator()
        self.setup_lt_evaluator()
        self.setup_bootstrapper()

    def setup_scheme(self, orion_params):
        self.NewScheme = LattigoFunction(
            self.lib.NewScheme,
            argtypes=[
                ctypes.c_int, 
                ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_char_p,
            ],
            restype=None
        )

        self.DeleteScheme = LattigoFunction(
            self.lib.DeleteScheme,
            argtypes=None,
            restype=None
        )

        runtime_memory_stats = getattr(self.lib, "GetRuntimeMemoryStats", None)
        if runtime_memory_stats is not None:
            self.GetRuntimeMemoryStats = LattigoFunction(
                runtime_memory_stats,
                argtypes=[],
                restype=ArrayResultUInt64,
            )

        reset_operation_counters = getattr(self.lib, "ResetOperationCounters", None)
        if reset_operation_counters is not None:
            self.ResetOperationCounters = LattigoFunction(
                reset_operation_counters,
                argtypes=[],
                restype=None,
            )

        operation_counters = getattr(self.lib, "GetOperationCounters", None)
        if operation_counters is not None:
            self.GetOperationCounters = LattigoFunction(
                operation_counters,
                argtypes=[],
                restype=ArrayResultUInt64,
            )

        self.FreeCArray = LattigoFunction(
            self.lib.FreeCArray,
            argtypes=[ctypes.c_void_p],
            restype=None
        )
        LattigoFunction.FreeCArray = self.FreeCArray

        logn = orion_params.get_logn()
        logq = orion_params.get_logq()
        logp = orion_params.get_logp()
        logscale = orion_params.get_logscale()
        h = orion_params.get_hamming_weight()
        ringtype = orion_params.get_ringtype()
        keys_path = orion_params.get_keys_path()
        io_mode = orion_params.get_io_mode()

        self.NewScheme(logn, logq, logp, logscale, h, ringtype, keys_path, io_mode)

    def setup_tensor_binds(self):
        self.DeletePlaintext = LattigoFunction(
            self.lib.DeletePlaintext,
            argtypes=[ctypes.c_int],
            restype=None
        )

        self.DeleteCiphertext = LattigoFunction(
            self.lib.DeleteCiphertext,
            argtypes=[ctypes.c_int],
            restype=None
        )

        self.GetPlaintextScale = LattigoFunction(
            self.lib.GetPlaintextScale,
            argtypes=[ctypes.c_int],
            restype=ctypes.c_ulong
        )

        self.GetPlaintextScaleLog2 = LattigoFunction(
            self.lib.GetPlaintextScaleLog2,
            argtypes=[ctypes.c_int],
            restype=ctypes.c_double
        )

        self.GetCiphertextScale = LattigoFunction(
            self.lib.GetCiphertextScale,
            argtypes=[ctypes.c_int],
            restype=ctypes.c_ulong
        )

        self.GetCiphertextScaleLog2 = LattigoFunction(
            self.lib.GetCiphertextScaleLog2,
            argtypes=[ctypes.c_int],
            restype=ctypes.c_double
        )

        self.SetPlaintextScale = LattigoFunction(
            self.lib.SetPlaintextScale,
            argtypes=[
                ctypes.c_int,
                ctypes.c_ulong,
            ],
            restype=None
        )

        self.SetCiphertextScale = LattigoFunction(
            self.lib.SetCiphertextScale,
            argtypes=[
                ctypes.c_int,
                ctypes.c_ulong,
            ],
            restype=None
        )

        self.GetPlaintextLevel = LattigoFunction(
            self.lib.GetPlaintextLevel,
            argtypes=[ctypes.c_int],
            restype=ctypes.c_int
        )
        
        self.GetCiphertextLevel = LattigoFunction(
            self.lib.GetCiphertextLevel,
            argtypes=[ctypes.c_int],
            restype=ctypes.c_int
        )

        self.GetPlaintextSlots = LattigoFunction(
            self.lib.GetPlaintextSlots,
            argtypes=[ctypes.c_int],
            restype=ctypes.c_int
        )
        
        self.GetCiphertextSlots = LattigoFunction(
            self.lib.GetCiphertextSlots,
            argtypes=[ctypes.c_int],
            restype=ctypes.c_int
        )

        self.GetCiphertextDegree = LattigoFunction(
            self.lib.GetCiphertextDegree,
            argtypes=[ctypes.c_int],
            restype=ctypes.c_int
        )

        self.GetModuliChain = LattigoFunction(
            self.lib.GetModuliChain,
            argtypes=None,
            restype=ArrayResultUInt64,
        )

        self.GetAuxModuliChain = LattigoFunction(
            self.lib.GetAuxModuliChain,
            argtypes=None,
            restype=ArrayResultUInt64,
        )

        self.GetLivePlaintexts = LattigoFunction(
            self.lib.GetLivePlaintexts,
            argtypes=None,
            restype=ArrayResultInt
        )

        self.GetLiveCiphertexts = LattigoFunction(
            self.lib.GetLiveCiphertexts,
            argtypes=None,
            restype=ArrayResultInt
        )

    def setup_key_generator(self):
        self.NewKeyGenerator = LattigoFunction(
            self.lib.NewKeyGenerator,
            argtypes=[],
            restype=None
        )

        self.GenerateSecretKey = LattigoFunction(
            self.lib.GenerateSecretKey,
            argtypes=[], 
            restype=None
        )

        self.GeneratePublicKey = LattigoFunction(
            self.lib.GeneratePublicKey,
            argtypes=[], 
            restype=None
        )

        self.GenerateRelinearizationKey = LattigoFunction(
            self.lib.GenerateRelinearizationKey,
            argtypes=[], 
            restype=None
        )

        self.GenerateEvaluationKeys = LattigoFunction(
            self.lib.GenerateEvaluationKeys,
            argtypes=[], 
            restype=None
        )

        self.SerializeSecretKey = LattigoFunction(
            self.lib.SerializeSecretKey,
            argtypes=[],
            restype=ArrayResultByte
        )

        self.LoadSecretKey = LattigoFunction(
            self.lib.LoadSecretKey,
            argtypes=[ctypes.POINTER(ctypes.c_ubyte), ctypes.c_ulong],
            restype=None
        )

    def setup_encoder(self):
        self.NewEncoder = LattigoFunction(
            self.lib.NewEncoder,
            argtypes=[],
            restype=None
        )

        self.Encode = LattigoFunction(
            self.lib.Encode,
            argtypes=[
                ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                ctypes.c_int,
                ctypes.c_ulong,
            ],
            restype=ctypes.c_int
        )
        self.Decode = LattigoFunction(
            self.lib.Decode,
            argtypes=[ctypes.c_int],
            restype=ArrayResultFloat,
        )
        self.DecodeComplex = LattigoFunction(
            self.lib.DecodeComplex,
            argtypes=[ctypes.c_int],
            restype=ArrayResultDouble,
        )

    def setup_encryptor(self):
        self.NewEncryptor = LattigoFunction(
            self.lib.NewEncryptor,
            argtypes=[],
            restype=None
        )

        self.NewDecryptor = LattigoFunction(
            self.lib.NewDecryptor,
            argtypes=[],
            restype=None
        )

        self.Encrypt = LattigoFunction(
            self.lib.Encrypt,
            argtypes=[ctypes.c_int],
            restype=ctypes.c_int
        )
        self.Decrypt = LattigoFunction(
            self.lib.Decrypt,
            argtypes=[ctypes.c_int],
            restype=ctypes.c_int
        )

    def setup_evaluator(self):
        self.NewEvaluator = LattigoFunction(
            self.lib.NewEvaluator,
            argtypes=[],
            restype=None
        )

        self.AddRotationKey = LattigoFunction(
            self.lib.AddRotationKey,
            argtypes=[ctypes.c_int],
            restype=None
        )

        self.Negate = LattigoFunction(
          self.lib.Negate,
            argtypes=[ctypes.c_int],
            restype=ctypes.c_int
        )

        self.Conjugate = LattigoFunction(
          self.lib.Conjugate,
            argtypes=[ctypes.c_int],
            restype=ctypes.c_int
        )

        self.ConjugateNew = LattigoFunction(
          self.lib.ConjugateNew,
            argtypes=[ctypes.c_int],
            restype=ctypes.c_int
        )

        self.Rotate = LattigoFunction(
          self.lib.Rotate,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.RotateNew = LattigoFunction(
            self.lib.RotateNew,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )       

        self.Rescale = LattigoFunction(
            self.lib.Rescale,
            argtypes=[ctypes.c_int],
            restype=ctypes.c_int
        )

        self.RescaleNew = LattigoFunction(
            self.lib.RescaleNew,
            argtypes=[ctypes.c_int],
            restype=ctypes.c_int
        )

        self.AddScalar = LattigoFunction(
            self.lib.AddScalar,
            argtypes=[
                ctypes.c_int,
                ctypes.c_float
            ],
            restype=ctypes.c_int
        )

        self.AddScalarNew = LattigoFunction(
            self.lib.AddScalarNew,
            argtypes=[
                ctypes.c_int,
                ctypes.c_float
            ],
            restype=ctypes.c_int
        )

        self.SubScalar = LattigoFunction(
            self.lib.SubScalar,
            argtypes=[
                ctypes.c_int,
                ctypes.c_float
            ],
            restype=ctypes.c_int
        )

        self.SubScalarNew = LattigoFunction(
            self.lib.SubScalarNew,
            argtypes=[
                ctypes.c_int,
                ctypes.c_float
            ],
            restype=ctypes.c_int
        )

        self.MulScalarInt = LattigoFunction(
            self.lib.MulScalarInt,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.MulScalarIntNew = LattigoFunction(
          self.lib.MulScalarIntNew,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.MulScalarFloat = LattigoFunction(
            self.lib.MulScalarFloat,
            argtypes=[
                ctypes.c_int,
                ctypes.c_float
            ],
            restype=ctypes.c_int
        )

        self.MulScalarFloatNew = LattigoFunction(
          self.lib.MulScalarFloatNew,
            argtypes=[
                ctypes.c_int,
                ctypes.c_float
            ],
            restype=ctypes.c_int
        )

        self.MulImaginaryUnit = LattigoFunction(
          self.lib.MulImaginaryUnit,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.MulImaginaryUnitNew = LattigoFunction(
          self.lib.MulImaginaryUnitNew,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.AddPlaintext = LattigoFunction(
          self.lib.AddPlaintext,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.AddPlaintextNew = LattigoFunction(
          self.lib.AddPlaintextNew,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.SubPlaintext = LattigoFunction(
          self.lib.SubPlaintext,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.SubPlaintextNew = LattigoFunction(
          self.lib.SubPlaintextNew,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.MulPlaintext = LattigoFunction(
          self.lib.MulPlaintext,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.MulPlaintextNew = LattigoFunction(
          self.lib.MulPlaintextNew,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.AddCiphertext = LattigoFunction(
          self.lib.AddCiphertext,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.AddCiphertextNew = LattigoFunction(
          self.lib.AddCiphertextNew,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.SubCiphertext = LattigoFunction(
          self.lib.SubCiphertext,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.SubCiphertextNew = LattigoFunction(
          self.lib.SubCiphertextNew,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.MulRelinCiphertext = LattigoFunction(
          self.lib.MulRelinCiphertext,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.MulRelinCiphertextNew = LattigoFunction(
          self.lib.MulRelinCiphertextNew,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

    def setup_poly_evaluator(self):
        self.NewPolynomialEvaluator = LattigoFunction(
            self.lib.NewPolynomialEvaluator,
            argtypes=[],
            restype=None
        )

        self.GenerateMonomial = LattigoFunction(
            self.lib.GenerateMonomial,
            argtypes=[ctypes.POINTER(ctypes.c_float), ctypes.c_int],
            restype=ctypes.c_int
        )

        self.GenerateChebyshev = LattigoFunction(
            self.lib.GenerateChebyshev,
            argtypes=[ctypes.POINTER(ctypes.c_float), ctypes.c_int],
            restype=ctypes.c_int
        )

        self.EvaluatePolynomial = LattigoFunction(
            self.lib.EvaluatePolynomial,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_ulong,
            ],
            restype=ctypes.c_int
        )

        self.GenerateMinimaxSignCoeffs = LattigoFunction(
            self.lib.GenerateMinimaxSignCoeffs,
            argtypes=[
                ctypes.POINTER(ctypes.c_int), ctypes.c_int, # degrees
                ctypes.c_int, # prec 
                ctypes.c_int, # logalpha
                ctypes.c_int, # logerr
                ctypes.c_int, # debug
            ],
            restype=ArrayResultDouble
        )

    def setup_lt_evaluator(self):
        self.NewLinearTransformEvaluator = LattigoFunction(
            self.lib.NewLinearTransformEvaluator,
            argtypes=[],
            restype=None
        )

        self.GenerateLinearTransform = LattigoFunction(
            self.lib.GenerateLinearTransform,
            argtypes=[
                ctypes.POINTER(ctypes.c_int), ctypes.c_int, # diags_idxs
                ctypes.POINTER(ctypes.c_float), ctypes.c_int, # diags_data
                ctypes.c_int, # level
                ctypes.c_float, # bsgs_ratio
                ctypes.c_char_p, # io_mode
            ],
            restype=ctypes.c_int
        )

        generate_batch = getattr(self.lib, "GenerateLinearTransformsBatch", None)
        if generate_batch is not None:
            self.GenerateLinearTransformsBatch = LattigoFunction(
                generate_batch,
                argtypes=[
                    ctypes.c_int,  # numTransforms
                    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),  # diagIdxsArray
                    ctypes.POINTER(ctypes.c_int),  # diagIdxsLens
                    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),  # diagDataArray
                    ctypes.POINTER(ctypes.c_int),  # diagDataLens
                    ctypes.POINTER(ctypes.c_int),  # levels
                    ctypes.c_float,  # bsgsRatio
                    ctypes.c_char_p,  # ioMode
                ],
                restype=ArrayResultInt
            )

        self.EvaluateLinearTransform = LattigoFunction(
            self.lib.EvaluateLinearTransform,
            argtypes=[
                ctypes.c_int, # transform ID
                ctypes.c_int, # ctxt ID
            ],
            restype=ctypes.c_int
        )

        self.DeleteLinearTransform = LattigoFunction(
            self.lib.DeleteLinearTransform,
            argtypes=[ctypes.c_int],
            restype=None
        )

        self.GetLinearTransformRotationKeys = LattigoFunction(
            self.lib.GetLinearTransformRotationKeys,
            argtypes=[ctypes.c_int],
            restype=ArrayResultInt
        )

        get_empty_plaintext_keys = getattr(self.lib, "GetLinearTransformEmptyPlaintextKeys", None)
        if get_empty_plaintext_keys is not None:
            self.GetLinearTransformEmptyPlaintextKeys = LattigoFunction(
                get_empty_plaintext_keys,
                argtypes=[ctypes.c_int],
                restype=ArrayResultInt
            )

        self.GenerateLinearTransformRotationKey = LattigoFunction(
            self.lib.GenerateLinearTransformRotationKey,
            argtypes=[ctypes.c_int],
            restype=None
        )

        self.GenerateLinearTransformsUnified = LattigoFunction(
            self.lib.GenerateLinearTransformsUnified,
            argtypes=[
                ctypes.c_int,  # numTransforms
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),  # diagIdxsArray
                ctypes.POINTER(ctypes.c_int),  # diagIdxsLens
                ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),  # diagDataArray
                ctypes.POINTER(ctypes.c_int),  # diagDataLens
                ctypes.POINTER(ctypes.c_int),  # levels
            ],
            restype=ArrayResultInt
        )

        self.GenerateLinearTransformsUnifiedComplex = LattigoFunction(
            self.lib.GenerateLinearTransformsUnifiedComplex,
            argtypes=[
                ctypes.c_int,  # numTransforms
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),  # diagIdxsArray
                ctypes.POINTER(ctypes.c_int),  # diagIdxsLens
                ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # interleaved real/imag diagDataArray
                ctypes.POINTER(ctypes.c_int),  # diagDataLens
                ctypes.POINTER(ctypes.c_int),  # levels
            ],
            restype=ArrayResultInt
        )

        generate_unified_load = getattr(self.lib, "GenerateLinearTransformsUnifiedLoad", None)
        if generate_unified_load is not None:
            self.GenerateLinearTransformsUnifiedLoad = LattigoFunction(
                generate_unified_load,
                argtypes=[
                    ctypes.c_int,  # numTransforms
                    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),  # diagIdxsArray
                    ctypes.POINTER(ctypes.c_int),  # diagIdxsLens
                    ctypes.POINTER(ctypes.c_int),  # levels
                ],
                restype=ArrayResultInt
            )

        self.EvaluateLinearTransformsWithSharedCache = LattigoFunction(
            self.lib.EvaluateLinearTransformsWithSharedCache,
            argtypes=[
                ctypes.POINTER(ctypes.c_int),  # transformIDs
                ctypes.c_int,  # numTransforms
                ctypes.c_int,  # ctxtID
            ],
            restype=ArrayResultInt
        )

        evaluate_sources_add = getattr(self.lib, "EvaluateLinearTransformSourcesWithSharedCacheAdd", None)
        if evaluate_sources_add is not None:
            self.EvaluateLinearTransformSourcesWithSharedCacheAdd = LattigoFunction(
                evaluate_sources_add,
                argtypes=[
                    ctypes.POINTER(ctypes.c_int),  # ctxtIDs
                    ctypes.c_int,  # numSources
                    ctypes.POINTER(ctypes.c_int),  # transformIDs
                    ctypes.POINTER(ctypes.c_int),  # targetIDs
                    ctypes.POINTER(ctypes.c_int),  # groupOffsets
                    ctypes.c_int,  # numPartials
                    ctypes.c_int,  # numTargets
                ],
                restype=ArrayResultInt
            )

        enable_profile = getattr(self.lib, "EnableLinearTransformEvaluationProfile", None)
        reset_profile = getattr(self.lib, "ResetLinearTransformEvaluationProfile", None)
        get_profile = getattr(self.lib, "GetLinearTransformEvaluationProfileCounters", None)
        if enable_profile is not None and reset_profile is not None and get_profile is not None:
            self.EnableLinearTransformEvaluationProfile = LattigoFunction(
                enable_profile,
                argtypes=[ctypes.c_int],
                restype=None,
            )
            self.ResetLinearTransformEvaluationProfile = LattigoFunction(
                reset_profile,
                argtypes=[],
                restype=None,
            )
            self.GetLinearTransformEvaluationProfileCounters = LattigoFunction(
                get_profile,
                argtypes=[],
                restype=ArrayResultUInt64,
            )

        consume_shared_cache_profile = getattr(self.lib, "ConsumeSharedCacheEvalProfileSeconds", None)
        if consume_shared_cache_profile is not None:
            self.ConsumeSharedCacheEvalProfileSeconds = LattigoFunction(
                consume_shared_cache_profile,
                argtypes=[],
                restype=ArrayResultDouble,
            )

        transform_uses_streaming = getattr(self.lib, "LinearTransformUsesStreaming", None)
        if transform_uses_streaming is not None:
            self.LinearTransformUsesStreaming = LattigoFunction(
                transform_uses_streaming,
                argtypes=[ctypes.c_int],
                restype=ctypes.c_int,
            )

        self.GenerateAndSerializeRotationKey = LattigoFunction(
            self.lib.GenerateAndSerializeRotationKey,
            argtypes=[ctypes.c_int],
            restype=ArrayResultByte
        )
        
        self.LoadRotationKey = LattigoFunction(
            self.lib.LoadRotationKey,
            argtypes=[
                ctypes.POINTER(ctypes.c_ubyte), ctypes.c_ulong,
                ctypes.c_ulong,
            ],
            restype=None
        )

        predecode_rotation_key = getattr(self.lib, "PredecodeRotationKey", None)
        install_predecoded_rotation_key = getattr(self.lib, "InstallPredecodedRotationKey", None)
        remove_predecoded_rotation_keys = getattr(self.lib, "RemovePredecodedRotationKeys", None)
        if (
            predecode_rotation_key is not None
            and install_predecoded_rotation_key is not None
            and remove_predecoded_rotation_keys is not None
        ):
            self.PredecodeRotationKey = LattigoFunction(
                predecode_rotation_key,
                argtypes=[
                    ctypes.POINTER(ctypes.c_ubyte), ctypes.c_ulong,
                    ctypes.c_ulong,
                ],
                restype=None,
            )
            self.InstallPredecodedRotationKey = LattigoFunction(
                install_predecoded_rotation_key,
                argtypes=[ctypes.c_ulong],
                restype=ctypes.c_int,
            )
            self.RemovePredecodedRotationKeys = LattigoFunction(
                remove_predecoded_rotation_keys,
                argtypes=[],
                restype=None,
            )

        self.SerializeDiagonal = LattigoFunction(
            self.lib.SerializeDiagonal,
            argtypes=[
                ctypes.c_int, # transform id
                ctypes.c_int, # diag index
            ],
            restype=ArrayResultByte
        )

        self.LoadPlaintextDiagonal = LattigoFunction(
            self.lib.LoadPlaintextDiagonal,
            argtypes=[
                ctypes.POINTER(ctypes.c_ubyte), ctypes.c_ulong,
                ctypes.c_int,
                ctypes.c_ulong,
            ],
            restype=None
        )

        self.LoadPlaintextDiagonalsBatch = LattigoFunction(
            self.lib.LoadPlaintextDiagonalsBatch,
            argtypes=[
                ctypes.POINTER(ctypes.c_ubyte), ctypes.c_ulong,
                ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_int,
                ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_int,
                ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                ctypes.c_int,
            ],
            restype=None
        )

        predecode_plaintexts = getattr(self.lib, "PredecodePlaintextDiagonalsBatch", None)
        install_predecoded_plaintexts = getattr(self.lib, "InstallPredecodedPlaintextDiagonals", None)
        remove_predecoded_plaintexts = getattr(self.lib, "RemovePredecodedPlaintextDiagonals", None)
        if (
            predecode_plaintexts is not None
            and install_predecoded_plaintexts is not None
            and remove_predecoded_plaintexts is not None
        ):
            self.PredecodePlaintextDiagonalsBatch = LattigoFunction(
                predecode_plaintexts,
                argtypes=[
                    ctypes.POINTER(ctypes.c_ubyte), ctypes.c_ulong,
                    ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_int,
                    ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_int,
                    ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                    ctypes.c_int,
                ],
                restype=None,
            )
            self.InstallPredecodedPlaintextDiagonals = LattigoFunction(
                install_predecoded_plaintexts,
                argtypes=[ctypes.c_int],
                restype=ctypes.c_int,
            )
            self.RemovePredecodedPlaintextDiagonals = LattigoFunction(
                remove_predecoded_plaintexts,
                argtypes=[ctypes.c_int],
                restype=None,
            )

        self.saved_io_host_predecode_supported = bool(
            callable(getattr(self, "PredecodeRotationKey", None))
            or callable(getattr(self, "PredecodePlaintextDiagonalsBatch", None))
        )

        self.RemovePlaintextDiagonals = LattigoFunction(
            self.lib.RemovePlaintextDiagonals,
            argtypes=[ctypes.c_int],
            restype=None
        )

        get_plaintext_levels = getattr(self.lib, "GetLinearTransformPlaintextLevels", None)
        if get_plaintext_levels is not None:
            self.GetLinearTransformPlaintextLevels = LattigoFunction(
                get_plaintext_levels,
                argtypes=[ctypes.c_int],
                restype=ArrayResultInt,
            )

        self.RemoveRotationKeys = LattigoFunction(
            self.lib.RemoveRotationKeys,
            argtypes=[],
            restype=None,
        )

    def setup_bootstrapper(self):
        self.NewBootstrapper = LattigoFunction(
            self.lib.NewBootstrapper,
            argtypes=[
                ctypes.POINTER(ctypes.c_int), ctypes.c_int, # logPs
                ctypes.c_int, # slots
            ], 
            restype=None
        )

        self.Bootstrap = LattigoFunction(
            self.lib.Bootstrap,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int,
            ],
            restype=ctypes.c_int
        )

        bootstrap_many = getattr(self.lib, "BootstrapMany", None)
        if bootstrap_many is not None:
            self.BootstrapMany = LattigoFunction(
                bootstrap_many,
                argtypes=[
                    ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                    ctypes.c_int,
                ],
                restype=ArrayResultInt
            )

        enable_profile = getattr(self.lib, "EnableBootstrapProfile", None)
        reset_profile = getattr(self.lib, "ResetBootstrapProfile", None)
        get_profile = getattr(self.lib, "GetBootstrapProfileCounters", None)
        if enable_profile is not None and reset_profile is not None and get_profile is not None:
            self.EnableBootstrapProfile = LattigoFunction(
                enable_profile,
                argtypes=[ctypes.c_int],
                restype=None,
            )
            self.ResetBootstrapProfile = LattigoFunction(
                reset_profile,
                argtypes=[],
                restype=None,
            )
            self.GetBootstrapProfileCounters = LattigoFunction(
                get_profile,
                argtypes=[],
                restype=ArrayResultUInt64,
            )

        self.DeleteBootstrappers = LattigoFunction(
            self.lib.DeleteBootstrappers,
            argtypes=None,
            restype=None
        )


class ArrayResultInt(ctypes.Structure):
    _fields_ = [("Data", ctypes.POINTER(ctypes.c_int)), ("Length", ctypes.c_ulong)]

class ArrayResultFloat(ctypes.Structure):
    _fields_ = [("Data", ctypes.POINTER(ctypes.c_float)), ("Length", ctypes.c_ulong)]

class ArrayResultDouble(ctypes.Structure):
    _fields_ = [("Data", ctypes.POINTER(ctypes.c_double)), ("Length", ctypes.c_ulong)]

class ArrayResultUInt64(ctypes.Structure):
    _fields_ = [
        ("Data", ctypes.POINTER(ctypes.c_ulonglong)),
        ("Length", ctypes.c_ulonglong),
    ]

class ArrayResultByte(ctypes.Structure):
    _fields_ = [("Data", ctypes.POINTER(ctypes.c_char)), ("Length", ctypes.c_ulong)]
