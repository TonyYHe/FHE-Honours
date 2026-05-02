import ctypes
import os
import platform

from orion.backend.lattigo.bindings import (
    ArrayResultByte,
    ArrayResultDouble,
    ArrayResultInt,
    ArrayResultUInt64,
    LattigoFunction,
    LattigoLibrary,
)


class _LattigoMinimaxSignCoeffGenerator:
    """Sidecar minimax coeff generator used by the Cheddar backend.

    The sign coefficients are backend-independent cleartext constants. Cheddar
    does not provide this helper yet, so reuse Orion's existing Lattigo shared
    library for coefficient generation while keeping all encrypted execution on
    the Cheddar backend.
    """

    def __init__(self):
        if platform.system() == "Linux":
            lib_name = "lattigo-linux.so"
        elif platform.system() == "Darwin":
            if platform.machine().lower() in ("arm64", "aarch64"):
                lib_name = "lattigo-mac-arm64.dylib"
            else:
                lib_name = "lattigo-mac.dylib"
        elif platform.system() == "Windows":
            lib_name = "lattigo-windows.dll"
        else:
            raise RuntimeError("Unsupported platform")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        lattigo_dir = os.path.join(os.path.dirname(current_dir), "lattigo")
        lib_path = os.path.join(lattigo_dir, lib_name)
        self.lib = ctypes.CDLL(lib_path)
        self._generate = self.lib.GenerateMinimaxSignCoeffs
        self._generate.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._generate.restype = ArrayResultDouble
        self._free = self.lib.FreeCArray
        self._free.argtypes = [ctypes.c_void_p]
        self._free.restype = None

    def __call__(self, degrees, prec, logalpha, logerr, debug):
        degree_array = (ctypes.c_int * len(degrees))(*degrees)
        result = self._generate(
            degree_array,
            ctypes.c_int(len(degrees)),
            ctypes.c_int(prec),
            ctypes.c_int(logalpha),
            ctypes.c_int(logerr),
            ctypes.c_int(debug),
        )
        try:
            return [float(result.Data[i]) for i in range(int(result.Length))]
        finally:
            self._free(ctypes.cast(result.Data, ctypes.c_void_p))


class CheddarLibrary(LattigoLibrary):
    """ctypes loader for the Orion-side Cheddar compatibility wrapper."""

    def __init__(self):
        super().__init__()
        self.lt_outputs_are_rescaled = True
        self.align_addition_scales = True
        self.load_plaintext_diagonals_requires_payload = False
        self.supports_index_only_linear_transform_load = True
        self.saved_io_prefetch_requires_device_memory = True
        self.saved_io_device_prefetch_enabled = (
            os.environ.get("ORION_CHEDDAR_GPU_PREFETCH", "1").lower()
            not in ("0", "false", "no", "off")
        )
        self.memory_bounded_unified_transforms = True
        self.memory_bounded_unified_evaluate = True
        self.retain_unified_rotation_keys = True
        self.retain_unified_plaintexts = (
            os.environ.get("ORION_UNIFIED_LT_PLAINTEXT_RESIDENCY", "1").lower()
            not in ("0", "false", "no", "off")
        )
        self.prefer_encoded_plaintext_payload_cache = (
            os.environ.get("ORION_CHEDDAR_SAVE_PLAINTEXT_PAYLOADS", "1").lower()
            not in ("0", "false", "no", "off")
        )
        self.supports_streaming_encoded_plaintext_payload_cache = True
        encoded_payload_max = os.environ.get(
            "ORION_CHEDDAR_SAVE_PLAINTEXT_PAYLOAD_MAX_DEVICE_BYTES",
            str(4 * 1024**3),
        )
        try:
            self.encoded_plaintext_payload_max_device_bytes = int(encoded_payload_max)
        except ValueError:
            self.encoded_plaintext_payload_max_device_bytes = 4 * 1024**3

    def _load_library(self):
        try:
            if platform.system() == "Linux":
                lib_name = "cheddar-linux.so"
            elif platform.system() == "Darwin":
                if platform.machine().lower() in ("arm64", "aarch64"):
                    lib_name = "cheddar-mac-arm64.dylib"
                else:
                    lib_name = "cheddar-mac.dylib"
            elif platform.system() == "Windows":
                lib_name = "cheddar-windows.dll"
            else:
                raise RuntimeError("Unsupported platform")

            current_dir = os.path.dirname(os.path.abspath(__file__))
            lib_path = os.path.join(current_dir, lib_name)
            return ctypes.CDLL(lib_path)
        except OSError as e:
            raise RuntimeError(f"Failed to load Cheddar library: {e}")

    def setup_poly_evaluator(self):
        super().setup_poly_evaluator()
        self.GenerateMinimaxSignCoeffs = _LattigoMinimaxSignCoeffGenerator()

    def setup_lt_evaluator(self):
        super().setup_lt_evaluator()
        self.GetLinearTransformRotationKeyRequests = LattigoFunction(
            self.lib.GetLinearTransformRotationKeyRequests,
            argtypes=[ctypes.c_int],
            restype=ArrayResultInt,
        )
        self.GenerateLinearTransformRotationKeyAtLevel = LattigoFunction(
            self.lib.GenerateLinearTransformRotationKeyAtLevel,
            argtypes=[ctypes.c_int, ctypes.c_int],
            restype=None,
        )
        self.GenerateAndSerializeRotationKeyAtLevel = LattigoFunction(
            self.lib.GenerateAndSerializeRotationKeyAtLevel,
            argtypes=[ctypes.c_int, ctypes.c_int],
            restype=ArrayResultByte,
        )
        self.SerializeLinearTransformPlaintexts = LattigoFunction(
            self.lib.SerializeLinearTransformPlaintexts,
            argtypes=[ctypes.c_int],
            restype=ArrayResultByte,
        )
        self.LinearTransformUsesStreaming = LattigoFunction(
            self.lib.LinearTransformUsesStreaming,
            argtypes=[ctypes.c_int],
            restype=ctypes.c_int,
        )
        self.LoadLinearTransformPlaintexts = LattigoFunction(
            self.lib.LoadLinearTransformPlaintexts,
            argtypes=[
                ctypes.POINTER(ctypes.c_ubyte),
                ctypes.c_ulong,
                ctypes.c_int,
            ],
            restype=None,
        )
        generate_unified_load = getattr(self.lib, "GenerateLinearTransformsUnifiedLoad", None)
        if generate_unified_load is not None:
            self.GenerateLinearTransformsUnifiedLoad = LattigoFunction(
                generate_unified_load,
                argtypes=[
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.POINTER(ctypes.c_int),
                ],
                restype=ArrayResultInt,
            )
        self.ReleaseLinearTransformMatrix = LattigoFunction(
            self.lib.ReleaseLinearTransformMatrix,
            argtypes=[ctypes.c_int],
            restype=None,
        )
        self.LoadLinearTransformRotationKey = LattigoFunction(
            self.lib.LoadLinearTransformRotationKey,
            argtypes=[
                ctypes.POINTER(ctypes.c_ubyte),
                ctypes.c_ulong,
                ctypes.c_ulong,
                ctypes.c_int,
            ],
            restype=None,
        )
        self.RemoveLinearTransformRotationKeys = LattigoFunction(
            self.lib.RemoveLinearTransformRotationKeys,
            argtypes=[ctypes.c_int],
            restype=None,
        )
        self.GetDeviceMemoryInfo = LattigoFunction(
            self.lib.GetDeviceMemoryInfo,
            argtypes=[],
            restype=ArrayResultUInt64,
        )
        self.SynchronizeDevice = LattigoFunction(
            self.lib.SynchronizeDevice,
            argtypes=[],
            restype=None,
        )
        self.TrimDeviceMemoryPool = LattigoFunction(
            self.lib.TrimDeviceMemoryPool,
            argtypes=[ctypes.c_ulonglong],
            restype=None,
        )
        self.ConsumeDeviceMemoryTrimSeconds = LattigoFunction(
            self.lib.ConsumeDeviceMemoryTrimSeconds,
            argtypes=[],
            restype=ctypes.c_double,
        )
        self.ConsumeSharedCacheEvalProfileSeconds = LattigoFunction(
            self.lib.ConsumeSharedCacheEvalProfileSeconds,
            argtypes=[],
            restype=ArrayResultDouble,
        )
        self.PrepareLinearTransformsSharedCachePlan = LattigoFunction(
            self.lib.PrepareLinearTransformsSharedCachePlan,
            argtypes=[ctypes.POINTER(ctypes.c_int), ctypes.c_int],
            restype=None,
        )
        self.EstimateLinearTransformDeviceBytes = LattigoFunction(
            self.lib.EstimateLinearTransformDeviceBytes,
            argtypes=[ctypes.c_int],
            restype=ArrayResultUInt64,
        )
