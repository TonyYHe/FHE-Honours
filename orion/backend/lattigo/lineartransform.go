package main

import (
	"C"
	"math"
	"os"
	"runtime"
	"runtime/debug"
	"sort"
	"strconv"
	"sync"
	"unsafe"

	"github.com/realqhc/lattigo/v6/circuits/ckks/lintrans"
	commonlintrans "github.com/realqhc/lattigo/v6/circuits/common/lintrans"
	"github.com/realqhc/lattigo/v6/core/rlwe"
	"github.com/realqhc/lattigo/v6/ring"
	"github.com/realqhc/lattigo/v6/ring/ringqp"
	"github.com/realqhc/lattigo/v6/schemes/ckks"
)

var ltHeap = NewHeapAllocator()

type lattigoStreamingLTState struct {
	shell           lintrans.LinearTransformation
	slots           int
	diagData        []float32
	diagOffsetByKey map[int]int
	chunks          [][]int
	mu              sync.Mutex
}

var (
	ltStreamingMu     sync.Mutex
	ltStreamingStates = make(map[int]*lattigoStreamingLTState)
)

func ltCompileWorkerCount(n int) int {
	if n <= 1 {
		return 1
	}
	workers := runtime.GOMAXPROCS(0)
	if raw := os.Getenv("ORION_LATTIGO_COMPILE_WORKERS"); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
			workers = parsed
		}
	}
	if workers < 1 {
		workers = 1
	}
	if workers > n {
		workers = n
	}
	return workers
}

func unifiedNoBSGSEnabled() bool {
	raw := os.Getenv("ORION_LATTIGO_UNIFIED_NO_BSGS")
	return raw == "1" || raw == "true" || raw == "TRUE" || raw == "yes" || raw == "on"
}

func envBool(raw string) bool {
	return raw == "1" || raw == "true" || raw == "TRUE" || raw == "yes" || raw == "on"
}

func envOff(raw string) bool {
	return raw == "0" || raw == "false" || raw == "FALSE" || raw == "no" || raw == "off"
}

func lattigoStreamingLTMinPlaintexts() int {
	if raw := os.Getenv("ORION_LATTIGO_STREAMING_LT_MIN_PLAINTEXTS"); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
			return parsed
		}
	}
	return 8192
}

func lattigoStreamingLTChunkPlaintexts() int {
	if raw := os.Getenv("ORION_LATTIGO_STREAMING_LT_CHUNK_PLAINTEXTS"); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
			return parsed
		}
	}
	return 1024
}

func lattigoStreamingLTSharedTransformLimit() int {
	if raw := os.Getenv("ORION_LATTIGO_STREAMING_LT_SHARED_TRANSFORMS"); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
			return parsed
		}
	}
	return 2
}

func lattigoStreamingLTEnabled(ioMode string, plaintextCount int) bool {
	if ioMode != "none" || plaintextCount <= 0 {
		return false
	}
	raw := os.Getenv("ORION_LATTIGO_STREAMING_LT")
	if envOff(raw) {
		return false
	}
	if envBool(raw) || raw == "force" || raw == "always" {
		return true
	}
	if raw == "" {
		return false
	}
	return plaintextCount >= lattigoStreamingLTMinPlaintexts()
}

func sortedIntKeys[V any](values map[int]V) []int {
	keys := make([]int, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Ints(keys)
	return keys
}

func normalizeDiagIndex(key, slots int) int {
	return key & (slots - 1)
}

func addDiagOffsetAliases(offsets map[int]int, key, offset, slots int) {
	normalized := normalizeDiagIndex(key, slots)
	offsets[key] = offset
	offsets[normalized] = offset
	if normalized != 0 {
		offsets[normalized-slots] = offset
	}
}

func newLinearTransformationShell(param lintrans.Parameters, explicitN1 int) lintrans.LinearTransformation {
	slots := 1 << param.LogDimensions.Cols
	vec := make(map[int]ringqp.Poly)
	n1 := 0
	if param.LogBabyStepGiantStepRatio < 0 {
		for _, diag := range param.DiagonalsIndexList {
			vec[normalizeDiagIndex(diag, slots)] = ringqp.Poly{}
		}
	} else {
		n1 = explicitN1
		if n1 <= 0 {
			n1 = commonlintrans.FindBestBSGSRatio(
				param.DiagonalsIndexList,
				slots,
				param.LogBabyStepGiantStepRatio,
			)
		}
		index, _, _ := commonlintrans.BSGSIndex(param.DiagonalsIndexList, slots, n1)
		for j := range index {
			for _, k := range index[j] {
				vec[normalizeDiagIndex(j+k, slots)] = ringqp.Poly{}
			}
		}
	}

	return lintrans.LinearTransformation{
		MetaData: &rlwe.MetaData{
			PlaintextMetaData: rlwe.PlaintextMetaData{
				LogDimensions: param.LogDimensions,
				Scale:         param.Scale,
				IsBatched:     true,
			},
			CiphertextMetaData: rlwe.CiphertextMetaData{
				IsNTT:        true,
				IsMontgomery: true,
			},
		},
		LogBabyStepGiantStepRatio: param.LogBabyStepGiantStepRatio,
		N1:                        n1,
		LevelQ:                    param.LevelQ,
		LevelP:                    param.LevelP,
		Vec:                       vec,
	}
}

func chunkLinearTransformKeys(shell lintrans.LinearTransformation, chunkLimit int) [][]int {
	if chunkLimit <= 0 {
		chunkLimit = lattigoStreamingLTChunkPlaintexts()
	}
	appendChunked := func(chunks [][]int, keys []int) [][]int {
		if len(keys) == 0 {
			return chunks
		}
		for len(keys) > chunkLimit {
			chunk := append([]int(nil), keys[:chunkLimit]...)
			chunks = append(chunks, chunk)
			keys = keys[chunkLimit:]
		}
		return append(chunks, append([]int(nil), keys...))
	}

	if shell.N1 == 0 {
		return appendChunked(nil, sortedIntKeys(shell.Vec))
	}

	index, _, _ := commonlintrans.LinearTransformation(shell).BSGSIndex()
	giants := sortedIntKeys(index)
	chunks := make([][]int, 0, len(giants))
	current := make([]int, 0, chunkLimit)
	slots := 1 << shell.LogDimensions.Cols
	for _, giant := range giants {
		group := make([]int, 0, len(index[giant]))
		for _, baby := range index[giant] {
			group = append(group, normalizeDiagIndex(giant+baby, slots))
		}
		sort.Ints(group)
		if len(group) > chunkLimit {
			if len(current) > 0 {
				chunks = append(chunks, current)
				current = make([]int, 0, chunkLimit)
			}
			chunks = appendChunked(chunks, group)
			continue
		}
		if len(current) > 0 && len(current)+len(group) > chunkLimit {
			chunks = append(chunks, current)
			current = make([]int, 0, chunkLimit)
		}
		current = append(current, group...)
	}
	if len(current) > 0 {
		chunks = append(chunks, current)
	}
	return chunks
}

func newStreamingLTStateFromC(
	shell lintrans.LinearTransformation,
	diagIdxs []int,
	diagDataC *C.float,
	diagDataLen C.int,
) *lattigoStreamingLTState {
	slots := 1 << shell.LogDimensions.Cols
	expectedLen := len(diagIdxs) * slots
	if int(diagDataLen) != expectedLen {
		panic(
			"streaming linear transform received mismatched diagonal data length: expected=" +
				strconv.Itoa(expectedLen) + " actual=" + strconv.Itoa(int(diagDataLen)),
		)
	}

	raw := unsafe.Slice(diagDataC, int(diagDataLen))
	diagData := make([]float32, len(raw))
	for i, value := range raw {
		diagData[i] = float32(value)
	}

	offsets := make(map[int]int, len(diagIdxs)*3)
	for offset, diag := range diagIdxs {
		addDiagOffsetAliases(offsets, diag, offset, slots)
	}

	return &lattigoStreamingLTState{
		shell:           shell,
		slots:           slots,
		diagData:        diagData,
		diagOffsetByKey: offsets,
		chunks:          chunkLinearTransformKeys(shell, lattigoStreamingLTChunkPlaintexts()),
	}
}

func registerStreamingLTState(transformID int, state *lattigoStreamingLTState) {
	if state == nil {
		return
	}
	ltStreamingMu.Lock()
	ltStreamingStates[transformID] = state
	ltStreamingMu.Unlock()
}

func lookupStreamingLTState(transformID int) (*lattigoStreamingLTState, bool) {
	ltStreamingMu.Lock()
	state, ok := ltStreamingStates[transformID]
	ltStreamingMu.Unlock()
	return state, ok
}

func deleteStreamingLTState(transformID int) {
	ltStreamingMu.Lock()
	delete(ltStreamingStates, transformID)
	ltStreamingMu.Unlock()
}

func hasStreamingLTState(transformID int) bool {
	_, ok := lookupStreamingLTState(transformID)
	return ok
}

func (state *lattigoStreamingLTState) chunkDiagonals(keys []int) lintrans.Diagonals[float64] {
	diagonals := make(lintrans.Diagonals[float64], len(keys))
	for _, key := range keys {
		offset, ok := state.diagOffsetByKey[key]
		if !ok {
			offset, ok = state.diagOffsetByKey[normalizeDiagIndex(key, state.slots)]
		}
		if !ok {
			panic("streaming linear transform missing diagonal key " + strconv.Itoa(key))
		}
		start := offset * state.slots
		values := make([]float64, state.slots)
		for i := 0; i < state.slots; i++ {
			values[i] = float64(state.diagData[start+i])
		}
		diagonals[key] = values
	}
	return diagonals
}

func (state *lattigoStreamingLTState) encodeChunk(keys []int) lintrans.LinearTransformation {
	ringQP := scheme.Params.RingQP().AtLevel(state.shell.LevelQ, state.shell.LevelP)
	vec := make(map[int]ringqp.Poly, len(keys))
	for _, key := range keys {
		vec[normalizeDiagIndex(key, state.slots)] = ringQP.NewPoly()
	}
	chunk := state.shell
	chunk.Vec = vec
	if err := lintrans.Encode(scheme.Encoder, state.chunkDiagonals(keys), chunk); err != nil {
		panic(err)
	}
	return chunk
}

func releaseStreamingLTChunkMemory(chunks []lintrans.LinearTransformation) {
	for i := range chunks {
		chunks[i].Vec = nil
	}
	runtime.GC()
	debug.FreeOSMemory()
}

func evaluateStreamingLinearTransformNew(
	transformID int,
	state *lattigoStreamingLTState,
	ctIn *rlwe.Ciphertext,
) (*rlwe.Ciphertext, error) {
	state.mu.Lock()
	defer state.mu.Unlock()

	var ctOut *rlwe.Ciphertext
	for _, keys := range state.chunks {
		chunk := state.encodeChunk(keys)
		ctChunk, err := scheme.LinEvaluator.EvaluateNew(ctIn, chunk)
		if err != nil {
			return nil, err
		}
		if ctOut == nil {
			ctOut = ctChunk
		} else {
			if err = scheme.Evaluator.Add(ctOut, ctChunk, ctOut); err != nil {
				return nil, err
			}
			ctChunk = nil
		}
		releaseStreamingLTChunkMemory([]lintrans.LinearTransformation{chunk})
	}
	if ctOut == nil {
		return nil, os.ErrInvalid
	}
	_ = transformID
	return ctOut, nil
}

func evaluateStreamingLinearTransformsWithSharedCacheNew(
	transformIDs []int,
	transforms []lintrans.LinearTransformation,
	ctIn *rlwe.Ciphertext,
) ([]*rlwe.Ciphertext, error) {
	n := len(transformIDs)
	outputs := make([]*rlwe.Ciphertext, n)
	streamingStates := make([]*lattigoStreamingLTState, n)
	maxChunks := 0
	for i, id := range transformIDs {
		if state, ok := lookupStreamingLTState(id); ok {
			state.mu.Lock()
			defer state.mu.Unlock()
			streamingStates[i] = state
			if len(state.chunks) > maxChunks {
				maxChunks = len(state.chunks)
			}
		}
	}

	plainTransforms := make([]lintrans.LinearTransformation, 0)
	plainIndices := make([]int, 0)
	for i, state := range streamingStates {
		if state != nil {
			continue
		}
		plainTransforms = append(plainTransforms, transforms[i])
		plainIndices = append(plainIndices, i)
	}
	if len(plainTransforms) > 0 {
		plainOutputs := make([]*rlwe.Ciphertext, len(plainTransforms))
		for i, transform := range plainTransforms {
			plainOutputs[i] = rlwe.NewCiphertext(*scheme.Params, 1, transform.LevelQ)
		}
		if err := scheme.LinEvaluator.EvaluateManyWithSharedCache(ctIn, plainTransforms, plainOutputs); err != nil {
			return nil, err
		}
		for i, outputIndex := range plainIndices {
			outputs[outputIndex] = plainOutputs[i]
		}
	}

	sharedLimit := lattigoStreamingLTSharedTransformLimit()
	if sharedLimit < 1 {
		sharedLimit = 1
	}
	for chunkIndex := 0; chunkIndex < maxChunks; chunkIndex++ {
		pendingTransforms := make([]lintrans.LinearTransformation, 0, sharedLimit)
		pendingIndices := make([]int, 0, sharedLimit)
		flush := func() error {
			if len(pendingTransforms) == 0 {
				return nil
			}
			chunkOutputs := make([]*rlwe.Ciphertext, len(pendingTransforms))
			for i, transform := range pendingTransforms {
				chunkOutputs[i] = rlwe.NewCiphertext(*scheme.Params, 1, transform.LevelQ)
			}
			if err := scheme.LinEvaluator.EvaluateManyWithSharedCache(ctIn, pendingTransforms, chunkOutputs); err != nil {
				return err
			}
			for i, outputIndex := range pendingIndices {
				if outputs[outputIndex] == nil {
					outputs[outputIndex] = chunkOutputs[i]
				} else {
					if err := scheme.Evaluator.Add(outputs[outputIndex], chunkOutputs[i], outputs[outputIndex]); err != nil {
						return err
					}
					chunkOutputs[i] = nil
				}
			}
			releaseStreamingLTChunkMemory(pendingTransforms)
			pendingTransforms = pendingTransforms[:0]
			pendingIndices = pendingIndices[:0]
			return nil
		}
		for transformIndex, state := range streamingStates {
			if state == nil || chunkIndex >= len(state.chunks) {
				continue
			}
			pendingTransforms = append(pendingTransforms, state.encodeChunk(state.chunks[chunkIndex]))
			pendingIndices = append(pendingIndices, transformIndex)
			if len(pendingTransforms) >= sharedLimit {
				if err := flush(); err != nil {
					return nil, err
				}
			}
		}
		if err := flush(); err != nil {
			return nil, err
		}
	}

	for i, output := range outputs {
		if output == nil {
			_ = i
			return nil, os.ErrInvalid
		}
	}
	return outputs, nil
}

func buildFloatDiagonalsFromC(
	diagIdxs []int,
	diagDataC *C.float,
	diagDataLen C.int,
	slots int,
) lintrans.Diagonals[float64] {
	expectedLen := len(diagIdxs) * slots
	if int(diagDataLen) != expectedLen {
		panic(
			"linear transform received mismatched diagonal data length: expected=" +
				strconv.Itoa(expectedLen) + " actual=" + strconv.Itoa(int(diagDataLen)),
		)
	}
	raw := unsafe.Slice(diagDataC, int(diagDataLen))
	diagonals := make(lintrans.Diagonals[float64], len(diagIdxs))
	for i, key := range diagIdxs {
		values := make([]float64, slots)
		start := i * slots
		for j := 0; j < slots; j++ {
			values[j] = float64(raw[start+j])
		}
		diagonals[key] = values
	}
	return diagonals
}

func encodeFloatTransformsParallel(
	diagonalsList []lintrans.Diagonals[float64],
	transforms []lintrans.LinearTransformation,
) error {
	n := len(transforms)
	if n == 0 {
		return nil
	}
	workers := ltCompileWorkerCount(n)
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
	workers := ltCompileWorkerCount(n)
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

func optimalUnifiedBSGSLogRatio(params []lintrans.Parameters) int {
	if len(params) == 0 {
		return 1
	}
	slots := 1 << params[0].LogDimensions.Cols
	diagonalSets := make([][]int, len(params))
	for i, param := range params {
		diagonalSets[i] = param.DiagonalsIndexList
	}
	optimalN1, _ := commonlintrans.FindOptimalUnifiedBSGSRatio(diagonalSets, slots)
	logRatio := 0
	for lr := 0; lr < 16; lr++ {
		if (1 << lr) >= optimalN1 {
			logRatio = lr
			break
		}
	}
	return logRatio
}

func applyOptimalUnifiedBSGSRatio(params []lintrans.Parameters) {
	logRatio := optimalUnifiedBSGSLogRatio(params)
	for i := range params {
		params[i].LogBabyStepGiantStepRatio = logRatio
	}
}

func newUnifiedNoBSGSTransformations(params []lintrans.Parameters) []lintrans.LinearTransformation {
	transforms := make([]lintrans.LinearTransformation, len(params))
	for i, param := range params {
		param.LogBabyStepGiantStepRatio = -1
		transforms[i] = lintrans.NewTransformation(scheme.Params, param)
	}
	return transforms
}

func newUnifiedLoadTransformations(params []lintrans.Parameters) []lintrans.LinearTransformation {
	if len(params) == 0 {
		return nil
	}
	if unifiedNoBSGSEnabled() {
		transforms := make([]lintrans.LinearTransformation, len(params))
		for i, param := range params {
			param.LogBabyStepGiantStepRatio = -1
			transforms[i] = newLinearTransformationShell(param, 0)
		}
		return transforms
	}

	slots := 1 << params[0].LogDimensions.Cols
	diagonalSets := make([][]int, len(params))
	for i, param := range params {
		diagonalSets[i] = param.DiagonalsIndexList
	}

	optimalN1, _ := commonlintrans.FindOptimalUnifiedBSGSRatio(diagonalSets, slots)
	logRatio := optimalUnifiedBSGSLogRatio(params)
	transforms := make([]lintrans.LinearTransformation, len(params))
	for i, param := range params {
		vec := make(map[int]ringqp.Poly)
		index, _, _ := commonlintrans.BSGSIndex(param.DiagonalsIndexList, slots, optimalN1)
		for j := range index {
			for _, k := range index[j] {
				vec[j+k] = ringqp.Poly{}
			}
		}

		transforms[i] = lintrans.LinearTransformation{
			MetaData: &rlwe.MetaData{
				PlaintextMetaData: rlwe.PlaintextMetaData{
					LogDimensions: param.LogDimensions,
					Scale:         param.Scale,
					IsBatched:     true,
				},
				CiphertextMetaData: rlwe.CiphertextMetaData{
					IsNTT:        true,
					IsMontgomery: true,
				},
			},
			LogBabyStepGiantStepRatio: logRatio,
			N1:                        optimalN1,
			LevelQ:                    param.LevelQ,
			LevelP:                    param.LevelP,
			Vec:                       vec,
		}
	}
	return transforms
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
	deleteStreamingLTState(int(id))
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
	slots := scheme.Params.MaxSlots()

	ltparams := lintrans.Parameters{
		DiagonalsIndexList:        diagIdxs,
		LevelQ:                    int(level),
		LevelP:                    scheme.Params.MaxLevelP(),
		Scale:                     rlwe.NewScale(scheme.Params.Q()[int(level)]),
		LogDimensions:             ring.Dimensions{Rows: 0, Cols: scheme.Params.LogMaxSlots()},
		LogBabyStepGiantStepRatio: int(math.Log(float64(bsgsRatio))),
	}

	streamingShell := newLinearTransformationShell(ltparams, 0)
	useStreaming := lattigoStreamingLTEnabled(ioMode, len(streamingShell.Vec))
	var lt lintrans.LinearTransformation
	var streamingState *lattigoStreamingLTState
	if useStreaming {
		lt = streamingShell
		streamingState = newStreamingLTStateFromC(lt, diagIdxs, diagDataC, diagDataLen)
	} else {
		lt = lintrans.NewTransformation(scheme.Params, ltparams)
	}

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
	} else if !useStreaming { // otherwise, generate diagonals here.
		diagonals := buildFloatDiagonalsFromC(diagIdxs, diagDataC, diagDataLen, slots)
		if err := lintrans.Encode(scheme.Encoder, diagonals, lt); err != nil {
			panic(err)
		}
	}

	// Return reference to linear transform object we just created
	ltID := ltHeap.Add(lt)
	registerStreamingLTState(ltID, streamingState)
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
	streamingStates := make([]*lattigoStreamingLTState, n)
	diagIdxsList := make([][]int, n)
	paramsList := make([]lintrans.Parameters, n)
	shells := make([]lintrans.LinearTransformation, n)
	totalPlaintexts := 0

	for i := 0; i < n; i++ {
		diagIdxs := CArrayToSlice(diagIdxsArraySlice[i], diagIdxsLensSlice[i], convertCIntToInt)
		diagIdxsList[i] = diagIdxs

		ltparams := lintrans.Parameters{
			DiagonalsIndexList:        diagIdxs,
			LevelQ:                    int(levelsSlice[i]),
			LevelP:                    scheme.Params.MaxLevelP(),
			Scale:                     rlwe.NewScale(scheme.Params.Q()[int(levelsSlice[i])]),
			LogDimensions:             ring.Dimensions{Rows: 0, Cols: scheme.Params.LogMaxSlots()},
			LogBabyStepGiantStepRatio: int(math.Log(float64(bsgsRatio))),
		}
		paramsList[i] = ltparams
		shells[i] = newLinearTransformationShell(ltparams, 0)
		totalPlaintexts += len(shells[i].Vec)
	}

	streamingBatch := lattigoStreamingLTEnabled(ioMode, totalPlaintexts)
	streamingAny := false
	for i := 0; i < n; i++ {
		diagIdxs := diagIdxsList[i]
		ltparams := paramsList[i]
		streamingShell := shells[i]
		useStreaming := streamingBatch || lattigoStreamingLTEnabled(ioMode, len(streamingShell.Vec))
		if useStreaming {
			transforms[i] = streamingShell
			streamingStates[i] = newStreamingLTStateFromC(
				streamingShell,
				diagIdxs,
				diagDataArraySlice[i],
				diagDataLensSlice[i],
			)
			streamingAny = true
			continue
		}

		transforms[i] = lintrans.NewTransformation(scheme.Params, ltparams)
		if ioMode == "load" {
			transforms[i].Vec = make(map[int]ringqp.Poly)
			for _, diag := range diagIdxs {
				transforms[i].Vec[diag] = ringqp.Poly{}
			}
		} else {
			diagonalsList[i] = buildFloatDiagonalsFromC(
				diagIdxs,
				diagDataArraySlice[i],
				diagDataLensSlice[i],
				slots,
			)
		}
	}

	if ioMode != "load" {
		if streamingAny {
			for i := 0; i < n; i++ {
				if streamingStates[i] != nil {
					continue
				}
				if err := lintrans.Encode(scheme.Encoder, diagonalsList[i], transforms[i]); err != nil {
					panic(err)
				}
			}
		} else {
			if err := encodeFloatTransformsParallel(diagonalsList, transforms); err != nil {
				panic(err)
			}
		}
	}

	ids := make([]int, n)
	for i, lt := range transforms {
		ids[i] = AddLinearTransform(lt)
		registerStreamingLTState(ids[i], streamingStates[i])
	}
	return SliceToCArray(ids, convertIntToCInt)
}

//export EvaluateLinearTransform
func EvaluateLinearTransform(transformID, ctxtID C.int) C.int {
	transform := RetrieveLinearTransform(int(transformID))
	ctIn := RetrieveCiphertext(int(ctxtID))
	ensureLinearTransformRotationKeys(transform)

	// Update the linear transform evaluator to have the most
	// recent set of rotation keys.
	scheme.LinEvaluator = lintrans.NewEvaluator(
		scheme.Evaluator.WithKey(scheme.EvalKeys),
	)

	var ctOut *rlwe.Ciphertext
	var err error
	if streamingState, ok := lookupStreamingLTState(int(transformID)); ok {
		ctOut, err = evaluateStreamingLinearTransformNew(int(transformID), streamingState, ctIn)
	} else {
		validateLinearTransformPlaintextLevels(int(transformID), transform)
		ctOut, err = scheme.LinEvaluator.EvaluateNew(ctIn, transform)
	}
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
	diagIdxsList := make([][]int, n)

	for i := 0; i < n; i++ {
		diagIdxs := CArrayToSlice(diagIdxsArraySlice[i], diagIdxsLensSlice[i], convertCIntToInt)
		diagIdxsList[i] = diagIdxs

		params[i] = lintrans.Parameters{
			DiagonalsIndexList:        diagIdxs,
			LevelQ:                    int(levelsSlice[i]),
			LevelP:                    scheme.Params.MaxLevelP(),
			Scale:                     rlwe.NewScale(scheme.Params.Q()[int(levelsSlice[i])]),
			LogDimensions:             ring.Dimensions{Rows: 0, Cols: scheme.Params.LogMaxSlots()},
			LogBabyStepGiantStepRatio: 1,
		}
	}

	if !unifiedNoBSGSEnabled() {
		applyOptimalUnifiedBSGSRatio(params)
	}
	transforms := newUnifiedLoadTransformations(params)

	streamingAny := false
	totalPlaintexts := 0
	for _, transform := range transforms {
		totalPlaintexts += len(transform.Vec)
	}
	if lattigoStreamingLTEnabled("none", totalPlaintexts) {
		streamingAny = true
	} else {
		for _, transform := range transforms {
			if lattigoStreamingLTEnabled("none", len(transform.Vec)) {
				streamingAny = true
				break
			}
		}
	}

	streamingStates := make([]*lattigoStreamingLTState, n)
	if streamingAny {
		for i := range transforms {
			streamingStates[i] = newStreamingLTStateFromC(
				transforms[i],
				diagIdxsList[i],
				diagDataArraySlice[i],
				diagDataLensSlice[i],
			)
		}
	} else {
		if unifiedNoBSGSEnabled() {
			transforms = newUnifiedNoBSGSTransformations(params)
		} else {
			transforms = lintrans.NewTransformationsWithUnifiedBSGS(scheme.Params, params)
		}
		diagonalsList := make([]lintrans.Diagonals[float64], n)
		for i := 0; i < n; i++ {
			diagonalsList[i] = buildFloatDiagonalsFromC(
				diagIdxsList[i],
				diagDataArraySlice[i],
				diagDataLensSlice[i],
				slots,
			)
		}
		if err := encodeUnifiedFloatTransformsParallel(diagonalsList, transforms); err != nil {
			panic(err)
		}
	}

	ids := make([]int, n)
	for i, lt := range transforms {
		ids[i] = AddLinearTransform(lt)
		registerStreamingLTState(ids[i], streamingStates[i])
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

	var transforms []lintrans.LinearTransformation
	if unifiedNoBSGSEnabled() {
		transforms = newUnifiedNoBSGSTransformations(params)
	} else {
		applyOptimalUnifiedBSGSRatio(params)
		transforms = lintrans.NewTransformationsWithUnifiedBSGS(scheme.Params, params)
	}
	if err := encodeUnifiedComplexTransformsParallel(diagonalsList, transforms); err != nil {
		panic(err)
	}

	ids := make([]int, n)
	for i, lt := range transforms {
		ids[i] = AddLinearTransform(lt)
	}
	return SliceToCArray(ids, convertIntToCInt)
}

//export GenerateLinearTransformsUnifiedLoad
func GenerateLinearTransformsUnifiedLoad(
	numTransforms C.int,
	diagIdxsArray **C.int, diagIdxsLens *C.int,
	levels *C.int,
) (*C.int, C.ulong) {
	n := int(numTransforms)
	diagIdxsArraySlice := unsafe.Slice(diagIdxsArray, n)
	diagIdxsLensSlice := unsafe.Slice(diagIdxsLens, n)
	levelsSlice := unsafe.Slice(levels, n)

	params := make([]lintrans.Parameters, n)
	for i := 0; i < n; i++ {
		diagIdxs := CArrayToSlice(diagIdxsArraySlice[i], diagIdxsLensSlice[i], convertCIntToInt)
		params[i] = lintrans.Parameters{
			DiagonalsIndexList:        diagIdxs,
			LevelQ:                    int(levelsSlice[i]),
			LevelP:                    scheme.Params.MaxLevelP(),
			Scale:                     rlwe.NewScale(scheme.Params.Q()[int(levelsSlice[i])]),
			LogDimensions:             ring.Dimensions{Rows: 0, Cols: scheme.Params.LogMaxSlots()},
			LogBabyStepGiantStepRatio: 1,
		}
	}

	transforms := newUnifiedLoadTransformations(params)
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
	streamingAny := false
	for i, id := range transformIDsSlice {
		transforms[i] = RetrieveLinearTransform(id)
		ensureLinearTransformRotationKeys(transforms[i])
		if hasStreamingLTState(id) {
			streamingAny = true
		} else {
			validateLinearTransformPlaintextLevels(id, transforms[i])
		}
	}

	ctIn := RetrieveCiphertext(int(ctxtID))
	scheme.LinEvaluator = lintrans.NewEvaluator(
		scheme.Evaluator.WithKey(scheme.EvalKeys),
	)

	if streamingAny {
		outputs, err := evaluateStreamingLinearTransformsWithSharedCacheNew(
			transformIDsSlice,
			transforms,
			ctIn,
		)
		if err != nil {
			panic(err)
		}
		outIDs := make([]int, n)
		for i, ct := range outputs {
			outIDs[i] = PushCiphertext(ct)
		}
		return SliceToCArray(outIDs, convertIntToCInt)
	}

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
