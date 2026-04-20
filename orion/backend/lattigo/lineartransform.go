package main

import (
	"C"
	"math"
	"unsafe"

	"github.com/realqhc/lattigo/v6/circuits/ckks/lintrans"
	"github.com/realqhc/lattigo/v6/core/rlwe"
	"github.com/realqhc/lattigo/v6/ring"
	"github.com/realqhc/lattigo/v6/ring/ringqp"
	"github.com/realqhc/lattigo/v6/schemes/ckks"
)

var ltHeap = NewHeapAllocator()

func AddLinearTransform(lt lintrans.LinearTransformation) int {
	return ltHeap.Add(lt)
}

func RetrieveLinearTransform(id int) lintrans.LinearTransformation {
	return ltHeap.Retrieve(id).(lintrans.LinearTransformation)
}

//export DeleteLinearTransform
func DeleteLinearTransform(id C.int) {
	ltHeap.Delete(int(id))
}

//export NewLinearTransformEvaluator
func NewLinearTransformEvaluator() {
	scheme.LinEvaluator = lintrans.NewEvaluator(
		ckks.NewEvaluator(*scheme.Params, scheme.EvalKeys))
}

//export GenerateLinearTransform
func GenerateLinearTransform(
	diagIdxsC *C.int, diagIdxsLen C.int,
	diagDataC *C.float, diagDataLen C.int,
	level C.int,
	bsgsRatio C.float,
	ioModeC *C.char,
) C.int {
	ioMode := C.GoString(ioModeC)

	// Unload diags data
	diagIdxs := CArrayToSlice(diagIdxsC, diagIdxsLen, convertCIntToInt)
	diagDataFlat := CArrayToSlice(diagDataC, diagDataLen, convertCFloatToFloat)

	// diagDataFlat is a flattened array of length len(diagIdxs) * slots.
	// The first element in diagIdxs corresponds to the first [0, slots]
	// values in diagsDataFlat, and so on. We'll extract these into a
	// dictionary that can be passed to Lattigo's LinearTransform evaluator.
	slots := scheme.Params.MaxSlots()
	diagonals := make(lintrans.Diagonals[float64])

	for i, key := range diagIdxs {
		diagonals[key] = diagDataFlat[i*slots : (i+1)*slots]
	}

	ltparams := lintrans.Parameters{
		DiagonalsIndexList:        diagonals.DiagonalsIndexList(),
		LevelQ:                    int(level),
		LevelP:                    scheme.Params.MaxLevelP(),
		Scale:                     rlwe.NewScale(scheme.Params.Q()[int(level)]),
		LogDimensions:             ring.Dimensions{Rows: 0, Cols: scheme.Params.LogMaxSlots()},
		LogBabyStepGiantStepRatio: int(math.Log(float64(bsgsRatio))),
	}

	lt := lintrans.NewTransformation(scheme.Params, ltparams)

	// ---------------------------- //
	//  Diagonal Generation/Saving  //
	// ---------------------------- //

	// If ioMode is "load", then we expect the diagonals to have already been
	// generated and serialized, so there's no need to regenerate them here.
	// We do, however, still need to instantiate empty plaintext diagonals.
	if ioMode == "load" {
		lt.Vec = make(map[int]ringqp.Poly)
		for _, diag := range diagIdxs {
			lt.Vec[diag] = ringqp.Poly{}
		}
	} else { // otherwise, generate diagonals here.
		if err := lintrans.Encode(scheme.Encoder, diagonals, lt); err != nil {
			panic(err)
		}
	}

	// Return reference to linear transform object we just created
	ltID := ltHeap.Add(lt)
	return C.int(ltID)
}

//export EvaluateLinearTransform
func EvaluateLinearTransform(transformID, ctxtID C.int) C.int {
	transform := RetrieveLinearTransform(int(transformID))
	ctIn := RetrieveCiphertext(int(ctxtID))

	// Update the linear transform evaluator to have the most
	// recent set of rotation keys.
	scheme.LinEvaluator = lintrans.NewEvaluator(
		scheme.Evaluator.WithKey(scheme.EvalKeys),
	)

	ctOut, err := scheme.LinEvaluator.EvaluateNew(ctIn, transform)
	if err != nil {
		panic(err)
	}

	idx := PushCiphertext(ctOut)
	return C.int(idx)
}

//export GetLinearTransformRotationKeys
func GetLinearTransformRotationKeys(transformID C.int) (*C.int, C.ulong) {
	transform := RetrieveLinearTransform(int(transformID))
	galEls := transform.GaloisElements(scheme.Params)

	arrPtr, length := SliceToCArray(galEls, convertULongtoInt)
	return arrPtr, length
}

//export GenerateLinearTransformRotationKey
func GenerateLinearTransformRotationKey(galEl C.int) {
	rotKey := scheme.KeyGen.GenGaloisKeyNew(uint64(galEl), scheme.SecretKey)
	scheme.EvalKeys.GaloisKeys[uint64(galEl)] = rotKey
}

//export GenerateAndSerializeRotationKey
func GenerateAndSerializeRotationKey(galEl C.int) (*C.char, C.ulong) {
	rotKey := scheme.KeyGen.GenGaloisKeyNew(uint64(galEl), scheme.SecretKey)
	data, err := rotKey.MarshalBinary() // Marshal the key to binary
	if err != nil {
		panic(err)
	}

	arrPtr, length := SliceToCArray(data, convertByteToCChar)
	return arrPtr, length
}

//export LoadRotationKey
func LoadRotationKey(
	dataPtr *C.char, lenData C.ulong,
	galEl C.ulong,
) {
	rotKeySerial := CArrayToByteSlice(unsafe.Pointer(dataPtr), uint64(lenData))

	// Unmarshal the binary data into a GaloisKey
	var rotKey rlwe.GaloisKey
	if err := rotKey.UnmarshalBinary(rotKeySerial); err != nil {
		panic(err)
	}

	// Update our global map of evaluation keys to include what
	// we just loaded. This will eventually get used by the
	// current linear transform and then deleted from RAM.
	scheme.EvalKeys.GaloisKeys[uint64(galEl)] = &rotKey
}

//export SerializeDiagonal
func SerializeDiagonal(transformID, diagIdx C.int) (*C.char, C.ulong) {
	transform := RetrieveLinearTransform(int(transformID))
	diag := transform.Vec[int(diagIdx)]

	data, err := diag.MarshalBinary() // Marshal the diag to binary
	if err != nil {
		panic(err)
	}

	// Since it will be saved to disk, we can delete it from our
	// linear transform object and load it in dynamically at runtime
	transform.Vec[int(diagIdx)] = ringqp.Poly{}

	arrPtr, length := SliceToCArray(data, convertByteToCChar)
	return arrPtr, length
}

//export LoadPlaintextDiagonal
func LoadPlaintextDiagonal(
	dataPtr *C.char, lenData C.ulong,
	transformID C.int,
	diagIdx C.ulong,
) {
	transform := RetrieveLinearTransform(int(transformID))
	diagSerial := CArrayToByteSlice(unsafe.Pointer(dataPtr), uint64(lenData))

	var poly ringqp.Poly
	if err := poly.UnmarshalBinary(diagSerial); err != nil {
		panic(err)
	}
	transform.Vec[int(diagIdx)] = poly
}

//export RemovePlaintextDiagonals
func RemovePlaintextDiagonals(transformID C.int) {
	linTransf := RetrieveLinearTransform(int(transformID))
	for diag := range linTransf.Vec {
		linTransf.Vec[diag] = ringqp.Poly{}
	}
}

//export RemoveRotationKeys
func RemoveRotationKeys() {
	// We'll just update the linear transform evaluator to no longer have
	// access to the Galois keys it had before. GC should do the rest.
	scheme.EvalKeys = rlwe.NewMemEvaluationKeySet(scheme.RelinKey)
	scheme.LinEvaluator = lintrans.NewEvaluator(scheme.Evaluator.WithKey(
		scheme.EvalKeys,
	))
}

//export GenerateLinearTransformsUnified
func GenerateLinearTransformsUnified(
	numTransforms C.int,
	diagIdxsArray **C.int, diagIdxsLens *C.int,
	diagDataArray **C.float, diagDataLens *C.int,
	levels *C.int,
) (*C.int, C.ulong) {
	n := int(numTransforms)
	slots := scheme.Params.MaxSlots()

	diagIdxsArraySlice := unsafe.Slice(diagIdxsArray, n)
	diagIdxsLensSlice := unsafe.Slice(diagIdxsLens, n)
	diagDataArraySlice := unsafe.Slice(diagDataArray, n)
	diagDataLensSlice := unsafe.Slice(diagDataLens, n)
	levelsSlice := unsafe.Slice(levels, n)

	params := make([]lintrans.Parameters, n)
	diagonalsList := make([]lintrans.Diagonals[float64], n)

	for i := 0; i < n; i++ {
		diagIdxs := CArrayToSlice(diagIdxsArraySlice[i], diagIdxsLensSlice[i], convertCIntToInt)
		diagDataFlat := CArrayToSlice(diagDataArraySlice[i], diagDataLensSlice[i], convertCFloatToFloat)

		diagonals := make(lintrans.Diagonals[float64])
		for j, key := range diagIdxs {
			diagonals[key] = diagDataFlat[j*slots : (j+1)*slots]
		}
		diagonalsList[i] = diagonals

		params[i] = lintrans.Parameters{
			DiagonalsIndexList:        diagonals.DiagonalsIndexList(),
			LevelQ:                    int(levelsSlice[i]),
			LevelP:                    scheme.Params.MaxLevelP(),
			Scale:                     rlwe.NewScale(scheme.Params.Q()[int(levelsSlice[i])]),
			LogDimensions:             ring.Dimensions{Rows: 0, Cols: scheme.Params.LogMaxSlots()},
			LogBabyStepGiantStepRatio: 1,
		}
	}

	transforms := lintrans.NewTransformationsWithUnifiedBSGS(scheme.Params, params)
	for i := range transforms {
		if err := lintrans.Encode(scheme.Encoder, diagonalsList[i], transforms[i]); err != nil {
			panic(err)
		}
	}

	ids := make([]int, n)
	for i, lt := range transforms {
		ids[i] = AddLinearTransform(lt)
	}
	return SliceToCArray(ids, convertIntToCInt)
}

//export GenerateLinearTransformsUnifiedComplex
func GenerateLinearTransformsUnifiedComplex(
	numTransforms C.int,
	diagIdxsArray **C.int, diagIdxsLens *C.int,
	diagDataArray **C.double, diagDataLens *C.int,
	levels *C.int,
) (*C.int, C.ulong) {
	n := int(numTransforms)
	slots := scheme.Params.MaxSlots()

	diagIdxsArraySlice := unsafe.Slice(diagIdxsArray, n)
	diagIdxsLensSlice := unsafe.Slice(diagIdxsLens, n)
	diagDataArraySlice := unsafe.Slice(diagDataArray, n)
	diagDataLensSlice := unsafe.Slice(diagDataLens, n)
	levelsSlice := unsafe.Slice(levels, n)

	params := make([]lintrans.Parameters, n)
	diagonalsList := make([]lintrans.Diagonals[complex128], n)

	for i := 0; i < n; i++ {
		diagIdxs := CArrayToSlice(diagIdxsArraySlice[i], diagIdxsLensSlice[i], convertCIntToInt)
		raw := unsafe.Slice(diagDataArraySlice[i], int(diagDataLensSlice[i]))
		diagonals := make(lintrans.Diagonals[complex128])
		for j, key := range diagIdxs {
			start := j * slots * 2
			values := make([]complex128, slots)
			for k := 0; k < slots; k++ {
				values[k] = complex(float64(raw[start+2*k]), float64(raw[start+2*k+1]))
			}
			diagonals[key] = values
		}
		diagonalsList[i] = diagonals

		params[i] = lintrans.Parameters{
			DiagonalsIndexList:        diagonals.DiagonalsIndexList(),
			LevelQ:                    int(levelsSlice[i]),
			LevelP:                    scheme.Params.MaxLevelP(),
			Scale:                     rlwe.NewScale(scheme.Params.Q()[int(levelsSlice[i])]),
			LogDimensions:             ring.Dimensions{Rows: 0, Cols: scheme.Params.LogMaxSlots()},
			LogBabyStepGiantStepRatio: 1,
		}
	}

	transforms := lintrans.NewTransformationsWithUnifiedBSGS(scheme.Params, params)
	for i := range transforms {
		if err := lintrans.Encode(scheme.Encoder, diagonalsList[i], transforms[i]); err != nil {
			panic(err)
		}
	}

	ids := make([]int, n)
	for i, lt := range transforms {
		ids[i] = AddLinearTransform(lt)
	}
	return SliceToCArray(ids, convertIntToCInt)
}

//export EvaluateLinearTransformsWithSharedCache
func EvaluateLinearTransformsWithSharedCache(
	transformIDs *C.int, numTransforms C.int,
	ctxtID C.int,
) (*C.int, C.ulong) {
	n := int(numTransforms)
	transformIDsSlice := CArrayToSlice(transformIDs, numTransforms, convertCIntToInt)

	transforms := make([]lintrans.LinearTransformation, n)
	for i, id := range transformIDsSlice {
		transforms[i] = RetrieveLinearTransform(id)
	}

	ctIn := RetrieveCiphertext(int(ctxtID))
	scheme.LinEvaluator = lintrans.NewEvaluator(
		scheme.Evaluator.WithKey(scheme.EvalKeys),
	)

	outputs := make([]*rlwe.Ciphertext, n)
	for i := range outputs {
		outputs[i] = rlwe.NewCiphertext(*scheme.Params, 1, transforms[i].LevelQ)
	}

	if err := scheme.LinEvaluator.EvaluateManyWithSharedCache(ctIn, transforms, outputs); err != nil {
		panic(err)
	}

	outIDs := make([]int, n)
	for i, ct := range outputs {
		outIDs[i] = PushCiphertext(ct)
	}
	return SliceToCArray(outIDs, convertIntToCInt)
}
