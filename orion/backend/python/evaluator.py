class NewEvaluator:
    def __init__(self, scheme):
        self.backend = scheme.backend
        self.new_evaluator()

    def new_evaluator(self):
        self.backend.NewEvaluator()

    def add_rotation_key(self, amount: int):
        self.backend.AddRotationKey(amount)

    def negate(self, ctxt):
        return self.backend.Negate(ctxt)

    def conjugate(self, ctxt, in_place):
        if in_place:
            return self.backend.Conjugate(ctxt)
        return self.backend.ConjugateNew(ctxt)
    
    def rotate(self, ctxt, amount, in_place):
        if in_place:
            return self.backend.Rotate(ctxt, amount)
        return self.backend.RotateNew(ctxt, amount)

    def add_scalar(self, ctxt, scalar, in_place):
        if in_place:
            return self.backend.AddScalar(ctxt, float(scalar))
        return self.backend.AddScalarNew(ctxt, float(scalar))

    def sub_scalar(self, ctxt, scalar, in_place):
        if in_place:
            return self.backend.SubScalar(ctxt, float(scalar))
        return self.backend.SubScalarNew(ctxt, float(scalar))

    def mul_scalar(self, ctxt, scalar, in_place):
        if isinstance(scalar, float) and scalar.is_integer():
            scalar = int(scalar)  # (e.g., 1.00 -> 1)

        if isinstance(scalar, int):
            ct_out = (self.backend.MulScalarInt if in_place 
                      else self.backend.MulScalarIntNew)(ctxt, scalar)
        else:
            ct_out = (self.backend.MulScalarFloat if in_place 
                      else self.backend.MulScalarFloatNew)(ctxt, scalar)
            ct_out = self.backend.Rescale(ct_out)

        return ct_out

    def mul_imaginary_unit(self, ctxt, sign, in_place):
        if in_place:
            return self.backend.MulImaginaryUnit(ctxt, int(sign))
        return self.backend.MulImaginaryUnitNew(ctxt, int(sign))

    def _align_addend_scale(self, ctxt, other, *, plaintext: bool):
        if not bool(getattr(self.backend, "align_addition_scales", False)):
            return
        lhs_log = float(self.backend.GetCiphertextScaleLog2(ctxt))
        if plaintext:
            rhs_log = float(self.backend.GetPlaintextScaleLog2(other))
            set_scale = self.backend.SetPlaintextScale
        else:
            rhs_log = float(self.backend.GetCiphertextScaleLog2(other))
            set_scale = self.backend.SetCiphertextScale

        # Cheddar tracks the exact post-rescale scale. Orion/Lattigo code often
        # treats tiny prime-induced scale drift as the default CKKS scale, so
        # align metadata only when the two operands are already effectively
        # equivalent.
        if abs(lhs_log - rhs_log) > 1e-2:
            raise RuntimeError(
                f"Refusing to align addition scales with log2 gap "
                f"{abs(lhs_log - rhs_log):.6g}"
            )
        set_scale(other, self.backend.GetCiphertextScale(ctxt))
        
    def add_plaintext(self, ctxt, ptxt, in_place):
        self._align_addend_scale(ctxt, ptxt, plaintext=True)
        if in_place:
            return self.backend.AddPlaintext(ctxt, ptxt) 
        return self.backend.AddPlaintextNew(ctxt, ptxt) 

    def sub_plaintext(self, ctxt, ptxt, in_place):
        self._align_addend_scale(ctxt, ptxt, plaintext=True)
        if in_place:
            return self.backend.SubPlaintext(ctxt, ptxt) 
        return self.backend.SubPlaintextNew(ctxt, ptxt) 

    def mul_plaintext(self, ctxt, ptxt, in_place):
        if in_place: # ct_out = ctxt
            ct_out = self.backend.MulPlaintext(ctxt, ptxt)
        else:
            ct_out = self.backend.MulPlaintextNew(ctxt, ptxt) 
        
        return self.backend.Rescale(ct_out)

    def add_ciphertext(self, ctxt0, ctxt1, in_place):
        self._align_addend_scale(ctxt0, ctxt1, plaintext=False)
        if in_place:
            return self.backend.AddCiphertext(ctxt0, ctxt1)
        return self.backend.AddCiphertextNew(ctxt0, ctxt1)

    def sub_ciphertext(self, ctxt0, ctxt1, in_place):
        self._align_addend_scale(ctxt0, ctxt1, plaintext=False)
        if in_place:
            return self.backend.SubCiphertext(ctxt0, ctxt1)
        return self.backend.SubCiphertextNew(ctxt0, ctxt1)

    def mul_ciphertext(self, ctxt0, ctxt1, in_place):
        if in_place: # ct_out = ctxt
            ct_out = self.backend.MulRelinCiphertext(ctxt0, ctxt1)
        else:
            ct_out = self.backend.MulRelinCiphertextNew(ctxt0, ctxt1)
        
        return self.backend.Rescale(ct_out)
    
    def rescale(self, ctxt, in_place):
        if in_place:
            return self.backend.Rescale(ctxt)
        return self.backend.RescaleNew(ctxt)
    
    def get_live_plaintexts(self):
        return self.backend.GetLivePlaintexts() 

    def get_live_ciphertexts(self):
        return self.backend.GetLiveCiphertexts() 
