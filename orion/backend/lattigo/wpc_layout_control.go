package main

import "C"

import "github.com/realqhc/lattigo/v6/circuits/ckks/lintrans"

// Logical coefficient payload, not Go object overhead or process RSS.
func linearTransformPayloadStats(transform lintrans.LinearTransformation) []uint64 {
	var qBytes, pBytes uint64
	for _, poly := range transform.Vec {
		for _, limb := range poly.Q.Coeffs {
			qBytes += uint64(len(limb)) * 8
		}
		for _, limb := range poly.P.Coeffs {
			pBytes += uint64(len(limb)) * 8
		}
	}
	return []uint64{1, uint64(len(transform.Vec)), qBytes, pBytes, qBytes + pBytes}
}

//export GetLinearTransformPayloadStats
func GetLinearTransformPayloadStats(transformID C.int) (*C.ulonglong, C.ulonglong) {
	values := linearTransformPayloadStats(RetrieveLinearTransform(int(transformID)))
	ptr, length := SliceToCArray(values, convertUint64ToCULonglong)
	return ptr, C.ulonglong(length)
}

// Copying aligned channel-first groups is not a homomorphic arithmetic operation.
// Return independently owned ciphertexts: releasing concat cannot invalidate skip.
//
//export CopyWPCLayoutCiphertext
func CopyWPCLayoutCiphertext(ciphertextID C.int, level C.int) C.int {
	input := RetrieveCiphertext(int(ciphertextID))
	if int(level) < 0 || int(level) > input.Level() {
		panic("native concat cannot increase a ciphertext level")
	}
	output := input.CopyNew()
	// Match the CIPS permutation's output level without spending a multiply.
	output.Resize(output.Degree(), int(level))
	return C.int(PushCiphertext(output))
}
