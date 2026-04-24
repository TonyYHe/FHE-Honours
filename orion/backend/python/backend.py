from __future__ import annotations

import io
import math
import ctypes
from dataclasses import dataclass
from itertools import count

import numpy as np
import torch


@dataclass
class _TensorState:
    values: torch.Tensor
    level: int
    scale: float
    slots: int
    degree: int = 0


@dataclass
class _LinearTransformState:
    diag_indices: list[int]
    diagonals: list[torch.Tensor]
    level: int
    slots: int


@dataclass
class _PolynomialState:
    kind: str
    coeffs: list[float]


class PythonBackend:
    """
    Lightweight in-memory backend used for Orion development and unit tests.

    It mirrors the small subset of the Lattigo bindings that Orion's Python
    wrapper expects, but all values stay as plain tensors in memory. This lets
    us compile and exercise packing/runtime paths without the shared library.
    """

    def __init__(self, params) -> None:
        self.params = params
        self.load_plaintext_diagonals_requires_payload = True
        self.saved_io_prefetch_requires_device_memory = False
        self._ids = count(1)
        self._plaintexts: dict[int, _TensorState] = {}
        self._ciphertexts: dict[int, _TensorState] = {}
        self._linear_transforms: dict[int, _LinearTransformState] = {}
        self._polynomials: dict[int, _PolynomialState] = {}
        self._rotation_keys: set[int] = set()
        self._num_slots = int(params.get_slots())
        self._moduli_chain = [float(2 ** int(q)) for q in params.get_logq()]
        self._aux_moduli_chain = [float(2 ** int(q)) for q in params.get_boot_logp()]

    # ---------------- #
    # Lifecycle / keys #
    # ---------------- #

    def DeleteScheme(self) -> None:
        self._plaintexts.clear()
        self._ciphertexts.clear()
        self._linear_transforms.clear()
        self._polynomials.clear()
        self._rotation_keys.clear()

    def NewKeyGenerator(self) -> None:
        return

    def GenerateSecretKey(self) -> None:
        return

    def SerializeSecretKey(self):
        return np.array([], dtype=np.uint8), None

    def LoadSecretKey(self, _serialized) -> None:
        return

    def GeneratePublicKey(self) -> None:
        return

    def GenerateRelinearizationKey(self) -> None:
        return

    def GenerateEvaluationKeys(self) -> None:
        return

    def NewEncoder(self) -> None:
        return

    def NewEncryptor(self) -> None:
        return

    def NewDecryptor(self) -> None:
        return

    def NewEvaluator(self) -> None:
        return

    def NewLinearTransformEvaluator(self) -> None:
        return

    def NewPolynomialEvaluator(self) -> None:
        return

    def NewBootstrapper(self, _logp, _slots) -> None:
        return

    def DeleteBootstrappers(self) -> None:
        return

    # ------------------- #
    # Plain / cipher data #
    # ------------------- #

    def _next_id(self) -> int:
        return int(next(self._ids))

    def _clone_values(self, values) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            tensor = values.detach().clone()
        else:
            tensor = torch.tensor(values)
        if bool(torch.is_complex(tensor)):
            if tensor.dtype in (torch.complex64, torch.complex128):
                return tensor
            return tensor.to(dtype=torch.complex64)
        if tensor.dtype == torch.float64:
            return tensor
        return tensor.to(dtype=torch.float32)

    def _decode_complex_list(self, values: torch.Tensor) -> list[float]:
        tensor = self._clone_values(values).reshape(-1)
        out: list[float] = []
        if bool(torch.is_complex(tensor)):
            for value in tensor.tolist():
                out.extend((float(np.real(value)), float(np.imag(value))))
            return out
        for value in tensor.tolist():
            out.extend((float(value), 0.0))
        return out

    def _store_linear_transform(self, diag_indices: list[int], diagonals: list[torch.Tensor], level: int, slots: int) -> int:
        transform_id = self._next_id()
        self._linear_transforms[transform_id] = _LinearTransformState(
            diag_indices=[int(idx) for idx in diag_indices],
            diagonals=[self._clone_values(diag).reshape(-1) for diag in diagonals],
            level=int(level),
            slots=int(slots),
        )
        return int(transform_id)

    def _store_plaintext(self, values, level: int, scale: float) -> int:
        plaintext_id = self._next_id()
        tensor = self._clone_values(values).reshape(-1)
        self._plaintexts[plaintext_id] = _TensorState(
            values=tensor,
            level=int(level),
            scale=float(scale),
            slots=int(tensor.numel()),
            degree=0,
        )
        return plaintext_id

    def _store_ciphertext(self, values, level: int, scale: float, degree: int = 1) -> int:
        ciphertext_id = self._next_id()
        tensor = self._clone_values(values).reshape(-1)
        self._ciphertexts[ciphertext_id] = _TensorState(
            values=tensor,
            level=int(level),
            scale=float(scale),
            slots=int(tensor.numel()),
            degree=int(degree),
        )
        return ciphertext_id

    def _clone_plaintext(self, plaintext_id: int) -> _TensorState:
        state = self._plaintexts[int(plaintext_id)]
        return _TensorState(
            values=state.values.clone(),
            level=state.level,
            scale=state.scale,
            slots=state.slots,
            degree=state.degree,
        )

    def _clone_ciphertext(self, ciphertext_id: int) -> _TensorState:
        state = self._ciphertexts[int(ciphertext_id)]
        return _TensorState(
            values=state.values.clone(),
            level=state.level,
            scale=state.scale,
            slots=state.slots,
            degree=state.degree,
        )

    def Encode(self, values, level, scale) -> int:
        return self._store_plaintext(values, level, scale)

    def Decode(self, plaintext_id):
        return self._plaintexts[int(plaintext_id)].values.tolist()

    def DecodeComplex(self, plaintext_id):
        return self._decode_complex_list(self._plaintexts[int(plaintext_id)].values)

    def Encrypt(self, plaintext_id) -> int:
        state = self._clone_plaintext(int(plaintext_id))
        return self._store_ciphertext(state.values, state.level, state.scale, degree=1)

    def Decrypt(self, ciphertext_id) -> int:
        state = self._clone_ciphertext(int(ciphertext_id))
        return self._store_plaintext(state.values, state.level, state.scale)

    def DeletePlaintext(self, plaintext_id) -> None:
        self._plaintexts.pop(int(plaintext_id), None)

    def DeleteCiphertext(self, ciphertext_id) -> None:
        self._ciphertexts.pop(int(ciphertext_id), None)

    # -------- #
    # Metadata #
    # -------- #

    def GetModuliChain(self):
        return list(self._moduli_chain)

    def GetAuxModuliChain(self):
        return list(self._aux_moduli_chain)

    def GetPlaintextScale(self, plaintext_id):
        return float(self._plaintexts[int(plaintext_id)].scale)

    def GetPlaintextScaleLog2(self, plaintext_id):
        scale = float(self._plaintexts[int(plaintext_id)].scale)
        if scale <= 0:
            return float("-inf")
        return math.log2(scale)

    def SetPlaintextScale(self, plaintext_id, scale) -> None:
        self._plaintexts[int(plaintext_id)].scale = float(scale)

    def GetPlaintextLevel(self, plaintext_id):
        return int(self._plaintexts[int(plaintext_id)].level)

    def GetPlaintextSlots(self, plaintext_id):
        return int(self._plaintexts[int(plaintext_id)].slots)

    def GetCiphertextScale(self, ciphertext_id):
        return float(self._ciphertexts[int(ciphertext_id)].scale)

    def GetCiphertextScaleLog2(self, ciphertext_id):
        scale = float(self._ciphertexts[int(ciphertext_id)].scale)
        if scale <= 0:
            return float("-inf")
        return math.log2(scale)

    def SetCiphertextScale(self, ciphertext_id, scale) -> None:
        self._ciphertexts[int(ciphertext_id)].scale = float(scale)

    def GetCiphertextLevel(self, ciphertext_id):
        return int(self._ciphertexts[int(ciphertext_id)].level)

    def GetCiphertextSlots(self, ciphertext_id):
        return int(self._ciphertexts[int(ciphertext_id)].slots)

    def GetCiphertextDegree(self, ciphertext_id):
        return int(self._ciphertexts[int(ciphertext_id)].degree)

    def GetLivePlaintexts(self):
        return len(self._plaintexts)

    def GetLiveCiphertexts(self):
        return len(self._ciphertexts)

    # ---------- #
    # Arithmetic #
    # ---------- #

    def _binary_ct_op(self, lhs_id, rhs_id, op, *, in_place: bool):
        lhs = self._ciphertexts[int(lhs_id)]
        rhs = self._ciphertexts[int(rhs_id)]
        values = op(lhs.values, rhs.values)
        level = min(lhs.level, rhs.level)
        scale = max(lhs.scale, rhs.scale)
        degree = max(lhs.degree, rhs.degree)
        if in_place:
            lhs.values = values
            lhs.level = level
            lhs.scale = scale
            lhs.degree = degree
            return int(lhs_id)
        return self._store_ciphertext(values, level, scale, degree)

    def _binary_pt_op(self, lhs_id, rhs_id, op, *, in_place: bool):
        lhs = self._ciphertexts[int(lhs_id)]
        rhs = self._plaintexts[int(rhs_id)]
        values = op(lhs.values, rhs.values)
        level = min(lhs.level, rhs.level)
        scale = max(lhs.scale, rhs.scale)
        if in_place:
            lhs.values = values
            lhs.level = level
            lhs.scale = scale
            return int(lhs_id)
        return self._store_ciphertext(values, level, scale, lhs.degree)

    def _scalar_ct_op(self, lhs_id, scalar, op, *, in_place: bool):
        lhs = self._ciphertexts[int(lhs_id)]
        values = op(lhs.values, float(scalar))
        if in_place:
            lhs.values = values
            return int(lhs_id)
        return self._store_ciphertext(values, lhs.level, lhs.scale, lhs.degree)

    def Negate(self, ciphertext_id):
        state = self._ciphertexts[int(ciphertext_id)]
        return self._store_ciphertext(-state.values, state.level, state.scale, state.degree)

    def Conjugate(self, ciphertext_id):
        state = self._ciphertexts[int(ciphertext_id)]
        state.values = torch.conj(state.values)
        return int(ciphertext_id)

    def ConjugateNew(self, ciphertext_id):
        state = self._clone_ciphertext(int(ciphertext_id))
        return self._store_ciphertext(torch.conj(state.values), state.level, state.scale, state.degree)

    def Rotate(self, ciphertext_id, amount):
        state = self._ciphertexts[int(ciphertext_id)]
        state.values = torch.roll(state.values, shifts=int(amount))
        return int(ciphertext_id)

    def RotateNew(self, ciphertext_id, amount):
        state = self._ciphertexts[int(ciphertext_id)]
        values = torch.roll(state.values, shifts=int(amount))
        return self._store_ciphertext(values, state.level, state.scale, state.degree)

    def AddRotationKey(self, amount: int) -> None:
        self._rotation_keys.add(int(amount))

    def AddScalar(self, ciphertext_id, scalar):
        return self._scalar_ct_op(ciphertext_id, scalar, lambda x, s: x + s, in_place=True)

    def AddScalarNew(self, ciphertext_id, scalar):
        return self._scalar_ct_op(ciphertext_id, scalar, lambda x, s: x + s, in_place=False)

    def SubScalar(self, ciphertext_id, scalar):
        return self._scalar_ct_op(ciphertext_id, scalar, lambda x, s: x - s, in_place=True)

    def SubScalarNew(self, ciphertext_id, scalar):
        return self._scalar_ct_op(ciphertext_id, scalar, lambda x, s: x - s, in_place=False)

    def MulScalarInt(self, ciphertext_id, scalar):
        return self._scalar_ct_op(ciphertext_id, scalar, lambda x, s: x * s, in_place=True)

    def MulScalarIntNew(self, ciphertext_id, scalar):
        return self._scalar_ct_op(ciphertext_id, scalar, lambda x, s: x * s, in_place=False)

    def MulScalarFloat(self, ciphertext_id, scalar):
        return self._scalar_ct_op(ciphertext_id, scalar, lambda x, s: x * s, in_place=True)

    def MulScalarFloatNew(self, ciphertext_id, scalar):
        return self._scalar_ct_op(ciphertext_id, scalar, lambda x, s: x * s, in_place=False)

    def MulImaginaryUnit(self, ciphertext_id, sign):
        state = self._ciphertexts[int(ciphertext_id)]
        state.values = state.values * complex(0.0, int(sign))
        return int(ciphertext_id)

    def MulImaginaryUnitNew(self, ciphertext_id, sign):
        state = self._ciphertexts[int(ciphertext_id)]
        return self._store_ciphertext(state.values * complex(0.0, int(sign)), state.level, state.scale, state.degree)

    def AddPlaintext(self, ciphertext_id, plaintext_id):
        return self._binary_pt_op(ciphertext_id, plaintext_id, torch.add, in_place=True)

    def AddPlaintextNew(self, ciphertext_id, plaintext_id):
        return self._binary_pt_op(ciphertext_id, plaintext_id, torch.add, in_place=False)

    def SubPlaintext(self, ciphertext_id, plaintext_id):
        return self._binary_pt_op(ciphertext_id, plaintext_id, torch.sub, in_place=True)

    def SubPlaintextNew(self, ciphertext_id, plaintext_id):
        return self._binary_pt_op(ciphertext_id, plaintext_id, torch.sub, in_place=False)

    def MulPlaintext(self, ciphertext_id, plaintext_id):
        return self._binary_pt_op(ciphertext_id, plaintext_id, torch.mul, in_place=True)

    def MulPlaintextNew(self, ciphertext_id, plaintext_id):
        return self._binary_pt_op(ciphertext_id, plaintext_id, torch.mul, in_place=False)

    def AddCiphertext(self, lhs_id, rhs_id):
        return self._binary_ct_op(lhs_id, rhs_id, torch.add, in_place=True)

    def AddCiphertextNew(self, lhs_id, rhs_id):
        return self._binary_ct_op(lhs_id, rhs_id, torch.add, in_place=False)

    def SubCiphertext(self, lhs_id, rhs_id):
        return self._binary_ct_op(lhs_id, rhs_id, torch.sub, in_place=True)

    def SubCiphertextNew(self, lhs_id, rhs_id):
        return self._binary_ct_op(lhs_id, rhs_id, torch.sub, in_place=False)

    def MulRelinCiphertext(self, lhs_id, rhs_id):
        result_id = self._binary_ct_op(lhs_id, rhs_id, torch.mul, in_place=True)
        self._ciphertexts[int(result_id)].degree = 1
        return result_id

    def MulRelinCiphertextNew(self, lhs_id, rhs_id):
        result_id = self._binary_ct_op(lhs_id, rhs_id, torch.mul, in_place=False)
        self._ciphertexts[int(result_id)].degree = 1
        return result_id

    def Rescale(self, ciphertext_id):
        state = self._ciphertexts[int(ciphertext_id)]
        state.level = max(0, int(state.level) - 1)
        return int(ciphertext_id)

    def RescaleNew(self, ciphertext_id):
        state = self._clone_ciphertext(int(ciphertext_id))
        state.level = max(0, int(state.level) - 1)
        return self._store_ciphertext(state.values, state.level, state.scale, state.degree)

    # ----------------- #
    # Linear transforms #
    # ----------------- #

    def GenerateLinearTransform(self, diags_idxs, diags_data, level, _bsgs_ratio, _io_mode):
        slots = int(len(diags_data) // max(1, len(diags_idxs))) if diags_idxs else self._num_slots
        diagonals = []
        cursor = 0
        for _ in diags_idxs:
            diagonals.append(self._clone_values(diags_data[cursor:cursor + slots]))
            cursor += slots
        return self._store_linear_transform([int(idx) for idx in diags_idxs], diagonals, int(level), int(slots))

    def GenerateLinearTransformsBatch(
        self,
        num_transforms,
        diag_idxs_ptrs,
        diag_idxs_lens,
        diag_data_ptrs,
        diag_data_lens,
        levels_array,
        _bsgs_ratio,
        _io_mode,
    ):
        out: list[int] = []
        for transform_index in range(int(num_transforms)):
            diag_len = int(diag_idxs_lens[transform_index])
            data_len = int(diag_data_lens[transform_index])
            diag_indices = [
                int(diag_idxs_ptrs[transform_index][diag_index])
                for diag_index in range(int(diag_len))
            ]
            slots = int(data_len // max(1, diag_len)) if diag_len else int(self._num_slots)
            diagonals: list[torch.Tensor] = []
            cursor = 0
            for _ in range(int(diag_len)):
                values = [
                    float(diag_data_ptrs[transform_index][cursor + offset])
                    for offset in range(int(slots))
                ]
                diagonals.append(torch.tensor(values, dtype=torch.float32))
                cursor += int(slots)
            out.append(
                self._store_linear_transform(
                    diag_indices,
                    diagonals,
                    int(levels_array[transform_index]),
                    int(slots),
                )
            )
        return out

    def GenerateLinearTransformsUnified(self, num_transforms, diag_idxs_ptrs, diag_idxs_lens, diag_data_ptrs, diag_data_lens, levels_array):
        out: list[int] = []
        count = int(num_transforms)
        for transform_index in range(int(count)):
            diag_len = int(diag_idxs_lens[transform_index])
            data_len = int(diag_data_lens[transform_index])
            diag_indices = [int(diag_idxs_ptrs[transform_index][diag_index]) for diag_index in range(int(diag_len))]
            slots = int(data_len // max(1, diag_len)) if diag_len else int(self._num_slots)
            diagonals: list[torch.Tensor] = []
            cursor = 0
            for _ in range(int(diag_len)):
                values = [float(diag_data_ptrs[transform_index][cursor + offset]) for offset in range(int(slots))]
                diagonals.append(torch.tensor(values, dtype=torch.float32))
                cursor += int(slots)
            out.append(self._store_linear_transform(diag_indices, diagonals, int(levels_array[transform_index]), int(slots)))
        return out

    def GenerateLinearTransformsUnifiedComplex(self, num_transforms, diag_idxs_ptrs, diag_idxs_lens, diag_data_ptrs, diag_data_lens, levels_array):
        out: list[int] = []
        count = int(num_transforms)
        for transform_index in range(int(count)):
            diag_len = int(diag_idxs_lens[transform_index])
            data_len = int(diag_data_lens[transform_index])
            diag_indices = [int(diag_idxs_ptrs[transform_index][diag_index]) for diag_index in range(int(diag_len))]
            slots = int(data_len // max(1, 2 * diag_len)) if diag_len else int(self._num_slots)
            diagonals: list[torch.Tensor] = []
            cursor = 0
            for _ in range(int(diag_len)):
                values: list[complex] = []
                for offset in range(int(slots)):
                    real = float(diag_data_ptrs[transform_index][cursor + 2 * offset])
                    imag = float(diag_data_ptrs[transform_index][cursor + 2 * offset + 1])
                    values.append(complex(real, imag))
                diagonals.append(torch.tensor(values, dtype=torch.complex64))
                cursor += int(2 * slots)
            out.append(self._store_linear_transform(diag_indices, diagonals, int(levels_array[transform_index]), int(slots)))
        return out

    def GetLinearTransformRotationKeys(self, transform_id):
        transform = self._linear_transforms[int(transform_id)]
        return [int(idx) for idx in transform.diag_indices]

    def GenerateLinearTransformRotationKey(self, key):
        self._rotation_keys.add(int(key))

    def GenerateAndSerializeRotationKey(self, key):
        self._rotation_keys.add(int(key))
        return np.array([int(key)], dtype=np.int32), None

    def FreeCArray(self, _ptr) -> None:
        return

    def SerializeDiagonal(self, transform_id, diag_idx):
        transform = self._linear_transforms[int(transform_id)]
        try:
            diag_pos = transform.diag_indices.index(int(diag_idx))
        except ValueError as exc:
            raise KeyError(
                f"Linear transform {transform_id} has no diagonal {diag_idx}"
            ) from exc

        buffer = io.BytesIO()
        np.save(
            buffer,
            transform.diagonals[int(diag_pos)].detach().cpu().numpy(),
            allow_pickle=False,
        )
        payload = np.frombuffer(buffer.getvalue(), dtype=np.uint8).copy()
        return payload, None

    def LoadRotationKey(self, _serialized_key) -> None:
        return

    def RemoveRotationKeys(self) -> None:
        return

    def LoadPlaintextDiagonal(self, serialized_diag, transform_id, diag_idx) -> None:
        transform = self._linear_transforms[int(transform_id)]
        buffer = io.BytesIO(np.asarray(serialized_diag, dtype=np.uint8).tobytes())
        diag = torch.from_numpy(np.load(buffer, allow_pickle=False)).reshape(-1).clone()
        try:
            diag_pos = transform.diag_indices.index(int(diag_idx))
        except ValueError:
            transform.diag_indices.append(int(diag_idx))
            transform.diagonals.append(diag)
        else:
            transform.diagonals[int(diag_pos)] = diag

    def LoadPlaintextDiagonalsBatch(self, payload, offsets, lengths, diag_indices, transform_id) -> None:
        payload_arr = np.asarray(payload, dtype=np.uint8).reshape(-1)
        for offset, length, diag_idx in zip(offsets, lengths, diag_indices):
            start = int(offset)
            end = int(start + length)
            self.LoadPlaintextDiagonal(
                payload_arr[start:end].copy(),
                int(transform_id),
                int(diag_idx),
            )

    def RemovePlaintextDiagonals(self, transform_id) -> None:
        transform = self._linear_transforms[int(transform_id)]
        transform.diagonals = [
            torch.zeros((0,), dtype=diag.dtype)
            for diag in transform.diagonals
        ]

    def EvaluateLinearTransform(self, transform_id, ciphertext_id):
        transform = self._linear_transforms[int(transform_id)]
        state = self._ciphertexts[int(ciphertext_id)]

        output_dtype = state.values.dtype
        for diag in transform.diagonals:
            output_dtype = torch.promote_types(output_dtype, diag.dtype)
        if output_dtype in (torch.complex64, torch.complex128):
            output_dtype = torch.complex128
        elif output_dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            output_dtype = torch.float64
        output = torch.zeros(transform.slots, dtype=output_dtype)
        for diag_idx, diag in zip(transform.diag_indices, transform.diagonals):
            rotated = torch.roll(state.values, shifts=-int(diag_idx))
            if output_dtype == torch.complex128:
                output += diag.to(dtype=torch.complex128) * rotated.to(dtype=torch.complex128)
            elif output_dtype == torch.float64:
                output += diag.to(dtype=torch.float64) * rotated.to(dtype=torch.float64)
            else:
                output += diag * rotated

        return self._store_ciphertext(
            output,
            min(state.level, transform.level),
            state.scale,
            state.degree,
        )

    def EvaluateLinearTransformsWithSharedCache(self, transform_ids_array, num_transforms, ct_input_id):
        return [
            int(self.EvaluateLinearTransform(int(transform_ids_array[index]), int(ct_input_id)))
            for index in range(int(num_transforms))
        ]

    def DeleteLinearTransform(self, transform_id) -> None:
        self._linear_transforms.pop(int(transform_id), None)

    # ---------- #
    # Polynomial #
    # ---------- #

    def GenerateMonomial(self, coeffs):
        poly_id = self._next_id()
        self._polynomials[poly_id] = _PolynomialState("monomial", [float(v) for v in coeffs])
        return poly_id

    def GenerateChebyshev(self, coeffs):
        poly_id = self._next_id()
        self._polynomials[poly_id] = _PolynomialState("chebyshev", [float(v) for v in coeffs])
        return poly_id

    def EvaluatePolynomial(self, ciphertext_id, poly_id, out_scale):
        state = self._ciphertexts[int(ciphertext_id)]
        poly = self._polynomials[int(poly_id)]

        if poly.kind == "monomial":
            values = torch.zeros_like(state.values)
            for coeff in poly.coeffs:
                values = values * state.values + float(coeff)
        else:
            coeffs = np.array(poly.coeffs, dtype=np.float32)
            values = torch.tensor(
                np.polynomial.chebyshev.chebval(state.values.numpy(), coeffs),
                dtype=torch.float32,
            )

        return self._store_ciphertext(values, state.level, float(out_scale), state.degree)

    def GenerateMinimaxSignCoeffs(self, _degrees, _prec, _logalpha, _logerr, _debug):
        raise NotImplementedError("Minimax sign coefficients are not implemented in the Python backend.")

    def GetPolyDepth(self, poly_id):
        coeff_count = max(1, len(self._polynomials[int(poly_id)].coeffs))
        return int(math.ceil(math.log2(coeff_count)))

    # ------------ #
    # Bootstrapping #
    # ------------ #

    def Bootstrap(self, ciphertext_id, _slots):
        state = self._clone_ciphertext(int(ciphertext_id))
        return self._store_ciphertext(
            state.values,
            self.params.get_max_level(),
            state.scale,
            state.degree,
        )
