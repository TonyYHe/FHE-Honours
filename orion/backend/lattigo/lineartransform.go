package main

import (
	"C"
	"math"
	"runtime"
	"strconv"
	"sync"
	"unsafe"

	"github.com/realqhc/lattigo/v6/circuits/ckks/lintrans"
	"github.com/realqhc/lattigo/v6/core/rlwe"
	"github.com/realqhc/lattigo/v6/ring"
	"github.com/realqhc/lattigo/v6/ring/ringqp"
	"github.com/realqhc/lattigo/v6/schemes/ckks"
)

var ltHeap = NewHeapAllocator()

func encodeFloatTransformsParallel(
	diagonalsList []lintrans.Diagonals[float64],
	transforms []lintrans.LinearTransformation,
) error {
	n := len(transforms)
	if n == 0 {
		return nil
	}
	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}
	if workers > n {
		workers = n
	}
	jobs := make(chan int, n)
	var wg sync.WaitGroup
	var once sync.Once
	var firstErr error

	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			encoder := ckks.NewEncoder(*scheme.Params)
			for idx := range jobs {
				if err := lintrans.Encode(encoder, diagonalsList[idx], transforms[idx]); err != nil {
					once.Do(func() {
						firstErr = err
					})
				}
			}
		}()
	}
	for i := 0; i < n; i++ {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	return firstErr
}

func encodeUnifiedFloatTransformsParallel(
	diagonalsList []lintrans.Diagonals[float64],
	transforms []lintrans.LinearTransformation,
) error {
	return encodeFloatTransformsParallel(diagonalsList, transforms)
}

func encodeUnifiedComplexTransformsParallel(
	diagonalsList []lintrans.Diagonals[complex128],
	transforms []lintrans.LinearTransformation,
) error {
	n := len(transforms)
	if n == 0 {
		return nil
	}
	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}
	if workers > n {
		workers = n
	}
	jobs := make(chan int, n)
	var wg sync.WaitGroup
	var once sync.Once
	var firstErr error

	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			encoder := ckks.NewEncoder(*scheme.Params)
			for idx := range jobs {
				if err := lintrans.Encode(encoder, diagonalsList[idx], transforms[idx]); err != nil {
					once.Do(func() {
						firstErr = err
					})
				}
			}
		}()
	}
	for i := 0; i < n; i++ {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	return firstErr
}

func AddLinearTransform(lt lintrans.LinearTransformation) int {
	return ltHeap.Add(lt)
}

func RetrieveLinearTransform(id int) lintrans.LinearTransformation {
	return ltHeap.Retrieve(id).(lintrans.LinearTransformation)
}

func ensureLinearTransformRotationKeys(transform lintrans.LinearTransformation) {
	if scheme.EvalKeys == nil {
		scheme.EvalKeys = rlwe.NewMemEvaluationKeySet(scheme.RelinKey)
	}
	if scheme.EvalKeys.GaloisKeys == nil {
		scheme.EvalKeys.GaloisKeys = make(map[uint64]*rlwe.GaloisKey)
	}
	for _, galEl := range transform.GaloisElements(scheme.Params) {
		if rotKey, exists := scheme.EvalKeys.GaloisKeys[galEl]; exists && rotKey != nil {
			continue
		}
		scheme.EvalKeys.GaloisKeys[galEl] = scheme.KeyGen.GenGaloisKeyNew(galEl, scheme.SecretKey)
	}
}

func emptyLinearTransformPlaintextKeys(transform lintrans.LinearTransformation) []int {
	keys := make([]int, 0)
	for diagIdx, poly := range transform.Vec {
		if len(poly.Q.Coeffs) == 0 {
			keys = append(keys, diagIdx)
		}
	}
	return keys
}

func linearTransformPlaintextLevels(transform lintrans.LinearTransformation) []int {
	levels := make([]int, 0, 2*len(transform.Vec))
	for diagIdx, poly := range transform.Vec {
		level := -1
		if len(poly.Q.Coeffs) > 0 {
			level = len(poly.Q.Coeffs) - 1
		}
		levels = append(levels, diagIdx, level)
	}
	return levels
}

func validateLinearTransformPlaintextLevels(transformID int, transform lintrans.LinearTransformation) {
	for diagIdx, poly := range transform.Vec {
		actualLevel := len(poly.Q.Coeffs) - 1
		if actualLevel < 0 {
			panic("linear transform plaintext missing: transformID=" + strconv.Itoa(transformID) + " diag=" + strconv.Itoa(diagIdx))
		}
		if actualLevel < transform.LevelQ {
			panic(
				"linear transform plaintext level mismatch: transformID=" + strconv.Itoa(transformID) +
					" diag=" + strconv.Itoa(diagIdx) +
					" transformLevel=" + strconv.Itoa(transform.LevelQ) +
					" plaintextLevel=" + strconv.Itoa(actualLevel),
			)
		}
	}
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
	diagDataFlat := []float64(nil)
	if ioMode != "load" {
		diagDataFlat = CArrayToSlice(diagDataC, diagDataLen, convertCFloatToFloat)
	}

	// diagDataFlat is a flattened array of length len(diagIdxs) * slots.
	// The first element in diagIdxs corresponds to the first [0, slots]
	// values in diagsDataFlat, and so on. We'll extract these into a
	// dictionary that can be passed to Lattigo's LinearTransform evaluator.
	slots := scheme.Params.MaxSlots()
	diagonals := make(lintrans.Diagonals[float64])

	if ioMode == "load" {
		for _, key := range diagIdxs {
			diagonals[key] = nil
		}
	} else {
		for i, key := range diagIdxs {
			diagonals[key] = diagDataFlat[i*slots : (i+1)*slots]
		}
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

//export GenerateLinearTransformsBatch
func GenerateLinearTransformsBatch(
	numTransforms C.int,
	diagIdxsArray **C.int, diagIdxsLens *C.int,
	diagDataArray **C.float, diagDataLens *C.int,
	levels *C.int,
	bsgsRatio C.float,
	ioModeC *C.char,
) (*C.int, C.ulong) {
	n := int(numTransforms)
	slots := scheme.Params.MaxSlots()
	ioMode := C.GoString(ioModeC)

	diagIdxsArraySlice := unsafe.Slice(diagIdxsArray, n)
	diagIdxsLensSlice := unsafe.Slice(diagIdxsLens, n)
	diagDataArraySlice := unsafe.Slice(diagDataArray, n)
	diagDataLensSlice := unsafe.Slice(diagDataLens, n)
	levelsSlice := unsafe.Slice(levels, n)

	transforms := make([]lintrans.LinearTransformation, n)
	diagonalsList := make([]lintrans.Diagonals[float64], n)

	for i := 0; i < n; i++ {
		diagIdxs := CArrayToSlice(diagIdxsArraySlice[i], diagIdxsLensSlice[i], convertCIntToInt)
		diagDataFlat := []float64(nil)
		if ioMode != "load" {
			diagDataFlat = CArrayToSlice(diagDataArraySlice[i], diagDataLensSlice[i], convertCFloatToFloat)
		}

		diagonals := make(lintrans.Diagonals[float64])
		if ioMode == "load" {
			for _, key := range diagIdxs {
				diagonals[key] = nil
			}
		} else {
			for j, key := range diagIdxs {
				diagonals[key] = diagDataFlat[j*slots : (j+1)*slots]
			}
		}
		diagonalsList[i] = diagonals

		ltparams := lintrans.Parameters{
			DiagonalsIndexList:        diagonals.DiagonalsIndexList(),
			LevelQ:                    int(levelsSlice[i]),
			LevelP:                    scheme.Params.MaxLevelP(),
			Scale:                     rlwe.NewScale(scheme.Params.Q()[int(levelsSlice[i])]),
			LogDimensions:             ring.Dimensions{Rows: 0, Cols: scheme.Params.LogMaxSlots()},
			LogBabyStepGiantStepRatio: int(math.Log(float64(bsgsRatio))),
		}
		transforms[i] = lintrans.NewTransformation(scheme.Params, ltparams)
		if ioMode == "load" {
			transforms[i].Vec = make(map[int]ringqp.Poly)
			for _, diag := range diagIdxs {
				transforms[i].Vec[diag] = ringqp.Poly{}
			}
		}
	}

	if ioMode != "load" {
		if err := encodeFloatTransformsParallel(diagonalsList, transforms); err != nil {
			panic(err)
		}
	}

	ids := make([]int, n)
	for i, lt := range transforms {
		ids[i] = AddLinearTransform(lt)
	}
	return SliceToCArray(ids, convertIntToCInt)
}

//export EvaluateLinearTransform
func EvaluateLinearTransform(transformID, ctxtID C.int) C.int {
	transform := RetrieveLinearTransform(int(transformID))
	ctIn := RetrieveCiphertext(int(ctxtID))
	ensureLinearTransformRotationKeys(transform)
	validateLinearTransformPlaintextLevels(int(transformID), transform)

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

//export GetLinearTransformEmptyPlaintextKeys
func GetLinearTransformEmptyPlaintextKeys(transformID C.int) (*C.int, C.ulong) {
	transform := RetrieveLinearTransform(int(transformID))
	keys := emptyLinearTransformPlaintextKeys(transform)
	arrPtr, length := SliceToCArray(keys, convertIntToCInt)
	return arrPtr, length
}

//export GetLinearTransformPlaintextLevels
func GetLinearTransformPlaintextLevels(transformID C.int) (*C.int, C.ulong) {
	transform := RetrieveLinearTransform(int(transformID))
	levels := linearTransformPlaintextLevels(transform)
	arrPtr, length := SliceToCArray(levels, convertIntToCInt)
	return arrPtr, length
}

//export GenerateLinearTransformRotationKey
func GenerateLinearTransformRotationKey(galEl C.int) {
	if _, exists := scheme.EvalKeys.GaloisKeys[uint64(galEl)]; exists {
		return
	}
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

//export LoadPlaintextDiagonalsBatch
func LoadPlaintextDiagonalsBatch(
	dataPtr *C.uchar, lenData C.ulong,
	offsetsPtr *C.ulonglong, offsetsLen C.int,
	lengthsPtr *C.ulonglong, lengthsLen C.int,
	diagIdxsPtr *C.int, diagIdxsLen C.int,
	transformID C.int,
) {
	if int(offsetsLen) != int(lengthsLen) || int(offsetsLen) != int(diagIdxsLen) {
		panic("LoadPlaintextDiagonalsBatch received mismatched batch array lengths")
	}

	transform := RetrieveLinearTransform(int(transformID))
	payload := CArrayToByteSlice(unsafe.Pointer(dataPtr), uint64(lenData))
	offsets := unsafe.Slice(offsetsPtr, int(offsetsLen))
	lengths := unsafe.Slice(lengthsPtr, int(lengthsLen))
	diagIdxs := unsafe.Slice(diagIdxsPtr, int(diagIdxsLen))

	for i := range diagIdxs {
		start := uint64(offsets[i])
		end := start + uint64(lengths[i])
		if end > uint64(len(payload)) {
			panic("LoadPlaintextDiagonalsBatch slice exceeds payload bounds")
		}

		var poly ringqp.Poly
		if err := poly.UnmarshalBinary(payload[start:end]); err != nil {
			panic(err)
		}
		transform.Vec[int(diagIdxs[i])] = poly
	}
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
	if err := encodeUnifiedFloatTransformsParallel(diagonalsList, transforms); err != nil {
		panic(err)
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
	if err := encodeUnifiedComplexTransformsParallel(diagonalsList, transforms); err != nil {
		panic(err)
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
		ensureLinearTransformRotationKeys(transforms[i])
		validateLinearTransformPlaintextLevels(id, transforms[i])
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
