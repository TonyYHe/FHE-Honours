package main

import (
	"C"

	"github.com/realqhc/lattigo/v6/core/rlwe"
	"github.com/realqhc/lattigo/v6/schemes/ckks"
)

//export NewEncoder
func NewEncoder() {
	scheme.Encoder = ckks.NewEncoder(*scheme.Params)
}

//export Encode
func Encode(
	valuesPtr *C.float,
	lenValues C.int,
	level C.int,
	scale C.ulong,
) C.int {
	values := CArrayToSlice(valuesPtr, lenValues, convertCFloatToFloat)
	plaintext := ckks.NewPlaintext(*scheme.Params, int(level))
	plaintext.Scale = rlwe.NewScale(uint64(scale))

	scheme.Encoder.Encode(values, plaintext)

	idx := PushPlaintext(plaintext)
	return C.int(idx)
}

//export Decode
func Decode(
	plaintextID C.int,
) (*C.float, C.ulong) {
	plaintext := RetrievePlaintext(int(plaintextID))
	result := make([]float64, scheme.Params.MaxSlots())
	scheme.Encoder.Decode(plaintext, result)

	arrPtr, length := SliceToCArray(result, convertFloatToCFloat)
	return arrPtr, length
}

//export DecodeComplex
func DecodeComplex(
	plaintextID C.int,
) (*C.double, C.ulong) {
	plaintext := RetrievePlaintext(int(plaintextID))
	result := make([]complex128, scheme.Params.MaxSlots())
	scheme.Encoder.Decode(plaintext, result)

	flat := make([]float64, 2*len(result))
	for i, value := range result {
		flat[2*i] = real(value)
		flat[2*i+1] = imag(value)
	}

	arrPtr, length := SliceToCArray(flat, convertFloat64ToCDouble)
	return arrPtr, length
}
