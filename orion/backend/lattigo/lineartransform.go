package main

import (
	"C"
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/realqhc/lattigo/v6/circuits/ckks/lintrans"
	commonlintrans "github.com/realqhc/lattigo/v6/circuits/common/lintrans"
	"github.com/realqhc/lattigo/v6/core/rlwe"
	"github.com/realqhc/lattigo/v6/ring"
	"github.com/realqhc/lattigo/v6/ring/ringqp"
	"github.com/realqhc/lattigo/v6/schemes/ckks"
	"github.com/realqhc/lattigo/v6/utils"
)

var ltHeap = NewHeapAllocator()

var (
	ltPredecodeMu             sync.Mutex
	ltPredecodedRotationKeys  = make(map[uint64]*rlwe.GaloisKey)
	ltPredecodedPlaintextVecs = make(map[int]map[int]ringqp.Poly)
	ltGOMAXPROCSMu            sync.Mutex
)

func clearPredecodedLinearTransformArtifacts() {
	ltPredecodeMu.Lock()
	defer ltPredecodeMu.Unlock()
	ltPredecodedRotationKeys = make(map[uint64]*rlwe.GaloisKey)
	ltPredecodedPlaintextVecs = make(map[int]map[int]ringqp.Poly)
}

func deletePredecodedPlaintextDiagonals(transformID int) {
	ltPredecodeMu.Lock()
	defer ltPredecodeMu.Unlock()
	delete(ltPredecodedPlaintextVecs, transformID)
}

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

type sharedCacheEvalProfile struct {
	planS              float64
	levelAdjustS       float64
	babyStepS          float64
	giantStepS         float64
	streamBuildMapS    float64
	streamEncodeHoistS float64
	streamLoadPayloadS float64
	streamEvalS        float64
	streamAccumulateS  float64
	pushS              float64
}

type linearTransformWrapperProfile struct {
	totalS             float64
	retrieveTransformS float64
	retrieveCipherS    float64
	ensureKeysS        float64
	newEvaluatorS      float64
	validateS          float64
	evaluateNewS       float64
	streamingEvaluateS float64
	pushS              float64
}

var (
	sharedCacheEvalProfileMu     sync.Mutex
	sharedCacheEvalProfileTotals sharedCacheEvalProfile
	ltWrapperProfileMu           sync.Mutex
	ltWrapperProfileTotals       linearTransformWrapperProfile
	ltWrapperProfileEnabled      bool
)

func accumulateSharedCacheEvalProfile(profile sharedCacheEvalProfile) {
	sharedCacheEvalProfileMu.Lock()
	sharedCacheEvalProfileTotals.planS += profile.planS
	sharedCacheEvalProfileTotals.levelAdjustS += profile.levelAdjustS
	sharedCacheEvalProfileTotals.babyStepS += profile.babyStepS
	sharedCacheEvalProfileTotals.giantStepS += profile.giantStepS
	sharedCacheEvalProfileTotals.streamBuildMapS += profile.streamBuildMapS
	sharedCacheEvalProfileTotals.streamEncodeHoistS += profile.streamEncodeHoistS
	sharedCacheEvalProfileTotals.streamLoadPayloadS += profile.streamLoadPayloadS
	sharedCacheEvalProfileTotals.streamEvalS += profile.streamEvalS
	sharedCacheEvalProfileTotals.streamAccumulateS += profile.streamAccumulateS
	sharedCacheEvalProfileTotals.pushS += profile.pushS
	sharedCacheEvalProfileMu.Unlock()
}

func recordSharedCacheEvalProfile(update func(*sharedCacheEvalProfile)) {
	sharedCacheEvalProfileMu.Lock()
	update(&sharedCacheEvalProfileTotals)
	sharedCacheEvalProfileMu.Unlock()
}

func resetSharedCacheEvalProfile() {
	sharedCacheEvalProfileMu.Lock()
	sharedCacheEvalProfileTotals = sharedCacheEvalProfile{}
	sharedCacheEvalProfileMu.Unlock()
}

func recordLTWrapperProfile(update func(*linearTransformWrapperProfile)) {
	ltWrapperProfileMu.Lock()
	if ltWrapperProfileEnabled {
		update(&ltWrapperProfileTotals)
	}
	ltWrapperProfileMu.Unlock()
}

func resetLTWrapperProfile() {
	ltWrapperProfileMu.Lock()
	ltWrapperProfileTotals = linearTransformWrapperProfile{}
	ltWrapperProfileMu.Unlock()
}

func setLTWrapperProfileEnabled(enabled bool) {
	ltWrapperProfileMu.Lock()
	ltWrapperProfileEnabled = enabled
	ltWrapperProfileMu.Unlock()
}

func secondsSince(started time.Time) float64 {
	return time.Since(started).Seconds()
}

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

func ltDiagonalEncodeWorkerCount(n int) int {
	if n <= 1 {
		return 1
	}
	workers := runtime.GOMAXPROCS(0)
	if raw := os.Getenv("ORION_SINGLE_SLOT_ENCODE_WORKERS"); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
			workers = parsed
		}
	} else if raw := os.Getenv("ORION_LATTIGO_DIAGONAL_ENCODE_WORKERS"); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
			workers = parsed
		}
	} else if raw := os.Getenv("ORION_LATTIGO_COMPILE_WORKERS"); raw != "" {
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

func withTemporaryGOMAXPROCS(workers int, fn func() error) error {
	if workers <= 0 {
		return fn()
	}
	ltGOMAXPROCSMu.Lock()
	defer ltGOMAXPROCSMu.Unlock()
	previous := runtime.GOMAXPROCS(workers)
	defer runtime.GOMAXPROCS(previous)
	return fn()
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

func positiveIntEnv(name string) (int, bool) {
	if raw := strings.TrimSpace(os.Getenv(name)); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
			return parsed, true
		}
	}
	return 0, false
}

func positiveFloatEnv(name string) (float64, bool) {
	if raw := strings.TrimSpace(os.Getenv(name)); raw != "" {
		if parsed, err := strconv.ParseFloat(raw, 64); err == nil && parsed > 0 {
			return parsed, true
		}
	}
	return 0, false
}

func lattigoStreamingLTMinPlaintexts() int {
	if raw := os.Getenv("ORION_LATTIGO_STREAMING_LT_MIN_PLAINTEXTS"); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
			return parsed
		}
	}
	return 8192
}

func hostMemoryInfoBytes() (total uint64, available uint64) {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0, 0
	}
	for _, line := range strings.Split(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		valueKB, err := strconv.ParseUint(fields[1], 10, 64)
		if err != nil {
			continue
		}
		switch strings.TrimSuffix(fields[0], ":") {
		case "MemTotal":
			total = valueKB * 1024
		case "MemAvailable":
			available = valueKB * 1024
		}
	}
	return total, available
}

func compileMemoryReserveBytes(total uint64) uint64 {
	if value, ok := positiveIntEnv("ORION_COMPILE_MEMORY_RESERVE_BYTES"); ok {
		return uint64(value)
	}
	if value, ok := positiveFloatEnv("ORION_COMPILE_MEMORY_RESERVE_GB"); ok {
		return uint64(value * float64(uint64(1)<<30))
	}
	fraction := 0.12
	if value, ok := positiveFloatEnv("ORION_COMPILE_MEMORY_RESERVE_FRACTION"); ok {
		fraction = math.Max(0, math.Min(0.95, value))
	}
	reserve := uint64(float64(total) * fraction)
	minReserve := uint64(16) << 30
	if reserve < minReserve {
		return minReserve
	}
	return reserve
}

func streamingLTAutoBudgetBytes() uint64 {
	total, available := hostMemoryInfoBytes()
	if available == 0 {
		return 0
	}
	reserve := compileMemoryReserveBytes(total)
	if available <= reserve {
		return 0
	}
	fraction := 0.75
	if value, ok := positiveFloatEnv("ORION_LATTIGO_STREAMING_LT_MEMORY_FRACTION"); ok {
		fraction = math.Max(0.05, math.Min(0.95, value))
	}
	return uint64(float64(available-reserve) * fraction)
}

func streamingLTMemoryOverhead() float64 {
	if value, ok := positiveFloatEnv("ORION_LATTIGO_STREAMING_LT_MEMORY_OVERHEAD"); ok {
		return math.Max(1.0, math.Min(16.0, value))
	}
	return 3.0
}

func streamingLTPlaintextBytes(levelQ, levelP int) uint64 {
	if scheme.Params == nil {
		return 0
	}
	if levelQ < 0 {
		levelQ = 0
	}
	if levelP < 0 {
		levelP = 0
	}
	moduli := uint64(levelQ + 1 + levelP + 1)
	degree := uint64(1) << uint(scheme.Params.LogN())
	return moduli * degree * uint64(8)
}

func floorPowerOfTwoInt(value uint64) int {
	if value <= 1 {
		return int(value)
	}
	power := uint64(1)
	for power <= value/2 {
		power *= 2
	}
	maxInt := uint64(^uint(0) >> 1)
	if power > maxInt {
		return int(maxInt)
	}
	return int(power)
}

func lattigoStreamingLTChunkPlaintextsFor(shell lintrans.LinearTransformation) int {
	if value, ok := positiveIntEnv("ORION_LATTIGO_STREAMING_LT_CHUNK_PLAINTEXTS"); ok {
		return value
	}
	minChunk := 512
	if value, ok := positiveIntEnv("ORION_LATTIGO_STREAMING_LT_CHUNK_PLAINTEXTS_MIN"); ok {
		minChunk = value
	}
	maxChunk := 4096
	if value, ok := positiveIntEnv("ORION_LATTIGO_STREAMING_LT_CHUNK_PLAINTEXTS_MAX"); ok {
		maxChunk = value
	}
	if maxChunk < minChunk {
		maxChunk = minChunk
	}
	plaintextBytes := streamingLTPlaintextBytes(shell.LevelQ, shell.LevelP)
	budget := streamingLTAutoBudgetBytes()
	sharedTarget := runtime.GOMAXPROCS(0)
	if value, ok := positiveIntEnv("ORION_LATTIGO_STREAMING_LT_SHARED_TRANSFORMS_MAX"); ok && value < sharedTarget {
		sharedTarget = value
	}
	if sharedTarget < 1 {
		sharedTarget = 1
	}
	if plaintextBytes == 0 || budget == 0 {
		if 1024 < minChunk {
			return minChunk
		}
		if 1024 > maxChunk {
			return maxChunk
		}
		return 1024
	}
	perPlaintext := uint64(float64(plaintextBytes) * streamingLTMemoryOverhead())
	if perPlaintext == 0 {
		perPlaintext = plaintextBytes
	}
	capacity := budget / (uint64(sharedTarget) * perPlaintext)
	chunk := floorPowerOfTwoInt(capacity)
	if chunk < minChunk {
		chunk = minChunk
	}
	if chunk > maxChunk {
		chunk = maxChunk
	}
	if chunk < 1 {
		chunk = 1
	}
	return chunk
}

func lattigoStreamingLTSharedTransformLimitFor(states []*lattigoStreamingLTState, chunkIndex int) int {
	activeCount := 0
	maxChunkBytes := uint64(0)
	for _, state := range states {
		if state == nil || chunkIndex < 0 || chunkIndex >= len(state.chunks) {
			continue
		}
		activeCount++
		plaintextBytes := streamingLTPlaintextBytes(state.shell.LevelQ, state.shell.LevelP)
		chunkBytes := plaintextBytes * uint64(len(state.chunks[chunkIndex]))
		if chunkBytes > maxChunkBytes {
			maxChunkBytes = chunkBytes
		}
	}
	if activeCount <= 0 {
		return 1
	}
	if value, ok := positiveIntEnv("ORION_LATTIGO_STREAMING_LT_SHARED_TRANSFORMS"); ok {
		return max(1, min(activeCount, value))
	}
	cpuLimit := runtime.GOMAXPROCS(0)
	if cpuLimit < 1 {
		cpuLimit = 1
	}
	if value, ok := positiveIntEnv("ORION_LATTIGO_STREAMING_LT_SHARED_TRANSFORMS_MAX"); ok {
		cpuLimit = min(cpuLimit, value)
	}
	limit := min(activeCount, cpuLimit)
	budget := streamingLTAutoBudgetBytes()
	if budget == 0 || maxChunkBytes == 0 {
		return max(1, limit)
	}
	perTransform := uint64(float64(maxChunkBytes) * streamingLTMemoryOverhead())
	if perTransform == 0 {
		perTransform = maxChunkBytes
	}
	memoryLimit := int(budget / perTransform)
	return max(1, min(limit, memoryLimit))
}

func lattigoStreamingLTChunkPlaintexts() int {
	if value, ok := positiveIntEnv("ORION_LATTIGO_STREAMING_LT_CHUNK_PLAINTEXTS"); ok {
		return value
	}
	return 1024
}

func lattigoStreamingLTEnabled(ioMode string, plaintextCount int) bool {
	if ioMode != "none" || plaintextCount <= 0 {
		return false
	}
	if !envBool(os.Getenv("ORION_LATTIGO_LEGACY_CHUNK_STREAMING_LT")) {
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
		chunkLimit = lattigoStreamingLTChunkPlaintextsFor(shell)
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
		chunks:          chunkLinearTransformKeys(shell, lattigoStreamingLTChunkPlaintextsFor(shell)),
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

func (state *lattigoStreamingLTState) newChunkShell(keys []int) lintrans.LinearTransformation {
	ringQP := scheme.Params.RingQP().AtLevel(state.shell.LevelQ, state.shell.LevelP)
	vec := make(map[int]ringqp.Poly, len(keys))
	for _, key := range keys {
		vec[normalizeDiagIndex(key, state.slots)] = ringQP.NewPoly()
	}
	chunk := state.shell
	chunk.Vec = vec
	return chunk
}

func encodeStreamingLTChunkBatch(
	states []*lattigoStreamingLTState,
	keyGroups [][]int,
	transforms []lintrans.LinearTransformation,
) {
	if len(transforms) != len(states) || len(transforms) != len(keyGroups) {
		panic(fmt.Errorf("streaming chunk batch length mismatch"))
	}
	if len(transforms) == 0 {
		return
	}
	if len(transforms) == 1 {
		if err := encodeSingleCKKSTransformDiagonalsParallel(states[0].chunkDiagonals(keyGroups[0]), transforms[0]); err != nil {
			panic(err)
		}
		return
	}
	diagonalsList := make([]lintrans.Diagonals[float64], len(transforms))
	for i, state := range states {
		diagonalsList[i] = state.chunkDiagonals(keyGroups[i])
	}
	if err := encodeCKKSTransformsDiagonalsParallel(diagonalsList, transforms); err != nil {
		panic(err)
	}
}

func (state *lattigoStreamingLTState) encodeChunk(keys []int) lintrans.LinearTransformation {
	chunk := state.newChunkShell(keys)
	if err := encodeSingleCKKSTransformDiagonalsParallel(state.chunkDiagonals(keys), chunk); err != nil {
		panic(err)
	}
	return chunk
}

func releaseStreamingLTChunkMemory(chunks []lintrans.LinearTransformation) {
	for i := range chunks {
		chunks[i].Vec = nil
	}
}

func evaluateStreamingLinearTransformNew(
	transformID int,
	state *lattigoStreamingLTState,
	ctIn *rlwe.Ciphertext,
	linEvaluator *lintrans.Evaluator,
) (*rlwe.Ciphertext, error) {
	state.mu.Lock()
	defer state.mu.Unlock()

	var ctOut *rlwe.Ciphertext
	for _, keys := range state.chunks {
		chunk := state.encodeChunk(keys)
		ctChunk, err := linEvaluator.EvaluateNew(ctIn, chunk)
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
	linEvaluator *lintrans.Evaluator,
) ([]*rlwe.Ciphertext, error) {
	profile := sharedCacheEvalProfile{}
	defer func() {
		accumulateSharedCacheEvalProfile(profile)
	}()

	planStarted := time.Now()
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
	profile.planS += secondsSince(planStarted)
	if len(plainTransforms) > 0 {
		plainOutputs := make([]*rlwe.Ciphertext, len(plainTransforms))
		for i, transform := range plainTransforms {
			plainOutputs[i] = rlwe.NewCiphertext(*scheme.Params, 1, transform.LevelQ)
		}
		evalStarted := time.Now()
		if err := linEvaluator.EvaluateManyWithSharedCache(ctIn, plainTransforms, plainOutputs); err != nil {
			return nil, err
		}
		profile.giantStepS += secondsSince(evalStarted)
		for i, outputIndex := range plainIndices {
			outputs[outputIndex] = plainOutputs[i]
		}
	}

	for chunkIndex := 0; chunkIndex < maxChunks; chunkIndex++ {
		sharedLimit := lattigoStreamingLTSharedTransformLimitFor(streamingStates, chunkIndex)
		if sharedLimit < 1 {
			sharedLimit = 1
		}
		pendingTransforms := make([]lintrans.LinearTransformation, 0, sharedLimit)
		pendingIndices := make([]int, 0, sharedLimit)
		pendingStates := make([]*lattigoStreamingLTState, 0, sharedLimit)
		pendingKeys := make([][]int, 0, sharedLimit)
		flush := func() error {
			if len(pendingTransforms) == 0 {
				return nil
			}
			encodeStarted := time.Now()
			encodeStreamingLTChunkBatch(pendingStates, pendingKeys, pendingTransforms)
			profile.streamEncodeHoistS += secondsSince(encodeStarted)
			chunkOutputs := make([]*rlwe.Ciphertext, len(pendingTransforms))
			for i, transform := range pendingTransforms {
				chunkOutputs[i] = rlwe.NewCiphertext(*scheme.Params, 1, transform.LevelQ)
			}
			evalStarted := time.Now()
			if err := linEvaluator.EvaluateManyWithSharedCache(ctIn, pendingTransforms, chunkOutputs); err != nil {
				return err
			}
			profile.streamEvalS += secondsSince(evalStarted)
			accumulateStarted := time.Now()
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
			profile.streamAccumulateS += secondsSince(accumulateStarted)
			releaseStreamingLTChunkMemory(pendingTransforms)
			pendingTransforms = pendingTransforms[:0]
			pendingIndices = pendingIndices[:0]
			pendingStates = pendingStates[:0]
			pendingKeys = pendingKeys[:0]
			return nil
		}
		for transformIndex, state := range streamingStates {
			if state == nil || chunkIndex >= len(state.chunks) {
				continue
			}
			keys := state.chunks[chunkIndex]
			buildStarted := time.Now()
			pendingTransforms = append(pendingTransforms, state.newChunkShell(keys))
			profile.streamBuildMapS += secondsSince(buildStarted)
			pendingIndices = append(pendingIndices, transformIndex)
			pendingStates = append(pendingStates, state)
			pendingKeys = append(pendingKeys, keys)
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

type diagonalEncodeJob struct {
	key int
	rot int
	vec ringqp.Poly
}

type diagonalEncodeBatchJob struct {
	transform int
	key       int
	rot       int
	vec       ringqp.Poly
}

func encodeSingleCKKSTransformDiagonalsParallel[T ckks.Float](
	diagonals lintrans.Diagonals[T],
	transform lintrans.LinearTransformation,
) error {
	commonTransform := commonlintrans.LinearTransformation(transform)
	commonDiagonals := commonlintrans.Diagonals[T](diagonals)
	rows := 1 << commonTransform.LogDimensions.Rows
	cols := 1 << commonTransform.LogDimensions.Cols

	jobs := make([]diagonalEncodeJob, 0, len(commonDiagonals))
	if commonTransform.N1 == 0 {
		for _, diagKey := range commonDiagonals.DiagonalsIndexList() {
			vecKey := diagKey
			if vecKey < 0 {
				vecKey += cols
			}
			vec, ok := commonTransform.Vec[vecKey]
			if !ok {
				return fmt.Errorf("cannot Encode: error encoding on LinearTransformation: plaintext diagonal [%d] does not exist", vecKey)
			}
			jobs = append(jobs, diagonalEncodeJob{key: diagKey, rot: 0, vec: vec})
		}
	} else {
		index, _, _ := commonTransform.BSGSIndex()
		for j := range index {
			rot := -j & (cols - 1)
			for _, i := range index[j] {
				diagKey := i + j
				vec, ok := commonTransform.Vec[diagKey]
				if !ok {
					return fmt.Errorf("cannot Encode: error encoding on LinearTransformation BSGS: input does not match the same non-zero diagonals")
				}
				jobs = append(jobs, diagonalEncodeJob{key: diagKey, rot: rot, vec: vec})
			}
		}
	}

	workers := ltDiagonalEncodeWorkerCount(len(jobs))
	if workers <= 1 {
		return lintrans.Encode(ckks.NewEncoder(*scheme.Params), diagonals, transform)
	}

	return withTemporaryGOMAXPROCS(workers, func() error {
		metaData := *commonTransform.MetaData
		metaData.Scale = commonTransform.Scale
		jobCh := make(chan diagonalEncodeJob, len(jobs))
		var wg sync.WaitGroup
		var once sync.Once
		var firstErr error

		for worker := 0; worker < workers; worker++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				encoder := ckks.NewEncoder(*scheme.Params)
				workerMetaData := metaData
				buf := make([]T, rows*cols)
				for job := range jobCh {
					values, err := commonDiagonals.At(job.key, cols)
					if err != nil {
						once.Do(func() {
							firstErr = fmt.Errorf("cannot Encode: %w", err)
						})
						continue
					}
					embedValues := values
					if job.rot != 0 {
						for row := 0; row < rows; row++ {
							start := row * cols
							end := start + cols
							utils.RotateSliceAllocFree(values[start:end], job.rot, buf[start:end])
						}
						embedValues = buf
					}
					if err := encoder.Embed(embedValues, &workerMetaData, job.vec); err != nil {
						once.Do(func() {
							firstErr = err
						})
					}
				}
			}()
		}

		for _, job := range jobs {
			jobCh <- job
		}
		close(jobCh)
		wg.Wait()
		return firstErr
	})
}

func encodeCKKSTransformsDiagonalsParallel[T ckks.Float](
	diagonalsList []lintrans.Diagonals[T],
	transforms []lintrans.LinearTransformation,
) error {
	n := len(transforms)
	if n == 0 {
		return nil
	}
	if n == 1 {
		return encodeSingleCKKSTransformDiagonalsParallel(diagonalsList[0], transforms[0])
	}
	commonTransforms := make([]commonlintrans.LinearTransformation, n)
	commonDiagonals := make([]commonlintrans.Diagonals[T], n)
	rowsByTransform := make([]int, n)
	colsByTransform := make([]int, n)
	jobs := make([]diagonalEncodeBatchJob, 0)
	maxValuesLen := 0
	for transformIndex := 0; transformIndex < n; transformIndex++ {
		commonTransform := commonlintrans.LinearTransformation(transforms[transformIndex])
		commonDiagonal := commonlintrans.Diagonals[T](diagonalsList[transformIndex])
		commonTransforms[transformIndex] = commonTransform
		commonDiagonals[transformIndex] = commonDiagonal
		rows := 1 << commonTransform.LogDimensions.Rows
		cols := 1 << commonTransform.LogDimensions.Cols
		rowsByTransform[transformIndex] = rows
		colsByTransform[transformIndex] = cols
		if rows*cols > maxValuesLen {
			maxValuesLen = rows * cols
		}
		if commonTransform.N1 == 0 {
			for _, diagKey := range commonDiagonal.DiagonalsIndexList() {
				vecKey := diagKey
				if vecKey < 0 {
					vecKey += cols
				}
				vec, ok := commonTransform.Vec[vecKey]
				if !ok {
					return fmt.Errorf("cannot Encode: error encoding on LinearTransformation: plaintext diagonal [%d] does not exist", vecKey)
				}
				jobs = append(jobs, diagonalEncodeBatchJob{
					transform: transformIndex,
					key:       diagKey,
					rot:       0,
					vec:       vec,
				})
			}
		} else {
			index, _, _ := commonTransform.BSGSIndex()
			for j := range index {
				rot := -j & (cols - 1)
				for _, i := range index[j] {
					diagKey := i + j
					vec, ok := commonTransform.Vec[diagKey]
					if !ok {
						return fmt.Errorf("cannot Encode: error encoding on LinearTransformation BSGS: input does not match the same non-zero diagonals")
					}
					jobs = append(jobs, diagonalEncodeBatchJob{
						transform: transformIndex,
						key:       diagKey,
						rot:       rot,
						vec:       vec,
					})
				}
			}
		}
	}

	workers := ltDiagonalEncodeWorkerCount(len(jobs))
	if workers <= 1 {
		for idx := 0; idx < n; idx++ {
			if err := encodeSingleCKKSTransformDiagonalsParallel(diagonalsList[idx], transforms[idx]); err != nil {
				return err
			}
		}
		return nil
	}

	return withTemporaryGOMAXPROCS(workers, func() error {
		jobCh := make(chan diagonalEncodeBatchJob, len(jobs))
		var wg sync.WaitGroup
		var once sync.Once
		var firstErr error

		for worker := 0; worker < workers; worker++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				encoder := ckks.NewEncoder(*scheme.Params)
				buf := make([]T, maxValuesLen)
				for job := range jobCh {
					transformIndex := int(job.transform)
					commonTransform := commonTransforms[transformIndex]
					rows := rowsByTransform[transformIndex]
					cols := colsByTransform[transformIndex]
					values, err := commonDiagonals[transformIndex].At(job.key, cols)
					if err != nil {
						once.Do(func() {
							firstErr = fmt.Errorf("cannot Encode: %w", err)
						})
						continue
					}
					embedValues := values
					if job.rot != 0 {
						for row := 0; row < rows; row++ {
							start := row * cols
							end := start + cols
							utils.RotateSliceAllocFree(values[start:end], job.rot, buf[start:end])
						}
						embedValues = buf[:rows*cols]
					}
					workerMetaData := *commonTransform.MetaData
					workerMetaData.Scale = commonTransform.Scale
					if err := encoder.Embed(embedValues, &workerMetaData, job.vec); err != nil {
						once.Do(func() {
							firstErr = err
						})
					}
				}
			}()
		}
		for _, job := range jobs {
			jobCh <- job
		}
		close(jobCh)
		wg.Wait()
		return firstErr
	})
}

func encodeFloatTransformsParallel(
	diagonalsList []lintrans.Diagonals[float64],
	transforms []lintrans.LinearTransformation,
) error {
	return encodeCKKSTransformsDiagonalsParallel(diagonalsList, transforms)
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
	return encodeCKKSTransformsDiagonalsParallel(diagonalsList, transforms)
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

func bumpLinearTransformEvalKeysVersion() {
	scheme.EvalKeysVersion++
	scheme.LinEvaluator = nil
	scheme.LinEvaluatorVersion = 0
}

func ensureLinearTransformEvalKeysMutable() bool {
	changed := false
	if scheme.EvalKeys == nil {
		scheme.EvalKeys = rlwe.NewMemEvaluationKeySet(scheme.RelinKey)
		changed = true
	}
	if scheme.EvalKeys.GaloisKeys == nil {
		scheme.EvalKeys.GaloisKeys = make(map[uint64]*rlwe.GaloisKey)
		changed = true
	}
	return changed
}

func newLinearTransformEvaluatorForEvalKeys() *lintrans.Evaluator {
	if scheme.Evaluator != nil {
		return lintrans.NewEvaluator(scheme.Evaluator.WithKey(scheme.EvalKeys))
	}
	return lintrans.NewEvaluator(ckks.NewEvaluator(*scheme.Params, scheme.EvalKeys))
}

func ensureCurrentLinearTransformEvaluator() *lintrans.Evaluator {
	if scheme.EvalKeys == nil {
		scheme.EvalKeys = rlwe.NewMemEvaluationKeySet(scheme.RelinKey)
		bumpLinearTransformEvalKeysVersion()
	}
	if scheme.LinEvaluator == nil || scheme.LinEvaluatorVersion != scheme.EvalKeysVersion {
		scheme.LinEvaluator = newLinearTransformEvaluatorForEvalKeys()
		scheme.LinEvaluatorVersion = scheme.EvalKeysVersion
	}
	return scheme.LinEvaluator
}

func ensureLinearTransformRotationKeys(transform lintrans.LinearTransformation) {
	changed := ensureLinearTransformEvalKeysMutable()
	for _, galEl := range transform.GaloisElements(scheme.Params) {
		if rotKey, exists := scheme.EvalKeys.GaloisKeys[galEl]; exists && rotKey != nil {
			continue
		}
		scheme.EvalKeys.GaloisKeys[galEl] = scheme.KeyGen.GenGaloisKeyNew(galEl, scheme.SecretKey)
		changed = true
	}
	if changed {
		bumpLinearTransformEvalKeysVersion()
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

//export EnableLinearTransformEvaluationProfile
func EnableLinearTransformEvaluationProfile(enabled C.int) {
	commonlintrans.EnableEvaluationProfile(int(enabled) != 0)
	setLTWrapperProfileEnabled(int(enabled) != 0)
}

//export ResetLinearTransformEvaluationProfile
func ResetLinearTransformEvaluationProfile() {
	commonlintrans.ResetEvaluationProfile()
	resetLTWrapperProfile()
}

//export GetLinearTransformEvaluationProfileCounters
func GetLinearTransformEvaluationProfileCounters() (*C.ulonglong, C.ulong) {
	profile := commonlintrans.GetEvaluationProfile()
	values := []uint64{
		uint64(profile.DiagonalTermCount),
		uint64(profile.QMulTermCount),
		uint64(profile.QPMulTermCount),
		uint64(profile.FinalModDownCount),
		uint64(profile.TransformCount),
		uint64(profile.BabyRotationCount),
		uint64(profile.GiantRotationCount),
		uint64(profile.InnerReduceCount),
		uint64(profile.OuterReduceCount),
	}
	return SliceToCArray(values, convertUint64ToCULonglong)
}

//export GetLinearTransformEvaluationProfileSeconds
func GetLinearTransformEvaluationProfileSeconds() (*C.double, C.ulong) {
	profile := commonlintrans.GetEvaluationProfile()
	ltWrapperProfileMu.Lock()
	wrapperProfile := ltWrapperProfileTotals
	ltWrapperProfileMu.Unlock()
	values := []float64{
		profile.SharedBufferSeconds,
		profile.DecomposeSeconds,
		profile.CollectRotationsSeconds,
		profile.PreRotateSeconds,
		profile.PreRotateAllocSeconds,
		profile.PreRotateAutomorphismSeconds,
		profile.TransformTotalSeconds,
		profile.TransformSetupSeconds,
		profile.TransformIndexSeconds,
		profile.TransformCopyScaleSeconds,
		profile.TransformMulAccumSeconds,
		profile.TransformInnerReduceSeconds,
		profile.TransformGiantModDownSeconds,
		profile.TransformGiantKeySwitchSeconds,
		profile.TransformGiantAutoSeconds,
		profile.TransformOuterReduceSeconds,
		profile.TransformFinalModDownSeconds,
		profile.TransformZeroDiagSeconds,
		profile.EvaluateManyTotalSeconds,
		profile.EvaluateManySetupSeconds,
		profile.EvaluateManyDecomposeSeconds,
		profile.EvaluateManyPreRotateSeconds,
		profile.EvaluateManyMultiplySeconds,
		wrapperProfile.totalS,
		wrapperProfile.retrieveTransformS,
		wrapperProfile.retrieveCipherS,
		wrapperProfile.ensureKeysS,
		wrapperProfile.newEvaluatorS,
		wrapperProfile.validateS,
		wrapperProfile.evaluateNewS,
		wrapperProfile.streamingEvaluateS,
		wrapperProfile.pushS,
	}
	return SliceToCArray(values, convertFloat64ToCDouble)
}

//export ConsumeSharedCacheEvalProfileSeconds
func ConsumeSharedCacheEvalProfileSeconds() (*C.double, C.ulong) {
	sharedCacheEvalProfileMu.Lock()
	profile := sharedCacheEvalProfileTotals
	sharedCacheEvalProfileTotals = sharedCacheEvalProfile{}
	sharedCacheEvalProfileMu.Unlock()
	values := []float64{
		profile.planS,
		profile.levelAdjustS,
		profile.babyStepS,
		profile.giantStepS,
		profile.streamBuildMapS,
		profile.streamEncodeHoistS,
		profile.streamLoadPayloadS,
		profile.streamEvalS,
		profile.streamAccumulateS,
		profile.pushS,
	}
	return SliceToCArray(values, convertFloat64ToCDouble)
}

//export LinearTransformUsesStreaming
func LinearTransformUsesStreaming(id C.int) C.int {
	if hasStreamingLTState(int(id)) {
		return C.int(1)
	}
	return C.int(0)
}

//export DeleteLinearTransform
func DeleteLinearTransform(id C.int) {
	deleteStreamingLTState(int(id))
	deletePredecodedPlaintextDiagonals(int(id))
	ltHeap.Delete(int(id))
}

//export GetLiveLinearTransformCount
func GetLiveLinearTransformCount() C.int {
	return C.int(len(ltHeap.GetLiveKeys()))
}

//export NewLinearTransformEvaluator
func NewLinearTransformEvaluator() {
	ensureCurrentLinearTransformEvaluator()
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
	wrapperStarted := time.Now()
	started := time.Now()
	transform := RetrieveLinearTransform(int(transformID))
	recordLTWrapperProfile(func(profile *linearTransformWrapperProfile) {
		profile.retrieveTransformS += secondsSince(started)
	})
	started = time.Now()
	ctIn := RetrieveCiphertext(int(ctxtID))
	recordLTWrapperProfile(func(profile *linearTransformWrapperProfile) {
		profile.retrieveCipherS += secondsSince(started)
	})
	started = time.Now()
	ensureLinearTransformRotationKeys(transform)
	recordLTWrapperProfile(func(profile *linearTransformWrapperProfile) {
		profile.ensureKeysS += secondsSince(started)
	})

	started = time.Now()
	linEvaluator := ensureCurrentLinearTransformEvaluator()
	recordLTWrapperProfile(func(profile *linearTransformWrapperProfile) {
		profile.newEvaluatorS += secondsSince(started)
	})

	var ctOut *rlwe.Ciphertext
	var err error
	if streamingState, ok := lookupStreamingLTState(int(transformID)); ok {
		started = time.Now()
		ctOut, err = evaluateStreamingLinearTransformNew(int(transformID), streamingState, ctIn, linEvaluator)
		recordLTWrapperProfile(func(profile *linearTransformWrapperProfile) {
			profile.streamingEvaluateS += secondsSince(started)
		})
	} else {
		started = time.Now()
		validateLinearTransformPlaintextLevels(int(transformID), transform)
		recordLTWrapperProfile(func(profile *linearTransformWrapperProfile) {
			profile.validateS += secondsSince(started)
		})
		started = time.Now()
		ctOut, err = linEvaluator.EvaluateNew(ctIn, transform)
		recordLTWrapperProfile(func(profile *linearTransformWrapperProfile) {
			profile.evaluateNewS += secondsSince(started)
		})
	}
	if err != nil {
		panic(err)
	}

	started = time.Now()
	idx := PushCiphertext(ctOut)
	recordLTWrapperProfile(func(profile *linearTransformWrapperProfile) {
		profile.pushS += secondsSince(started)
		profile.totalS += secondsSince(wrapperStarted)
	})
	return C.int(idx)
}

//export GetLinearTransformRotationKeys
func GetLinearTransformRotationKeys(transformID C.int) (*C.int, C.ulong) {
	transform := RetrieveLinearTransform(int(transformID))
	galEls := transform.GaloisElements(scheme.Params)

	arrPtr, length := SliceToCArray(galEls, convertULongtoInt)
	return arrPtr, length
}

//export GetLinearTransformRotationEvalCount
func GetLinearTransformRotationEvalCount(transformID C.int) C.int {
	transform := RetrieveLinearTransform(int(transformID))
	count := 0
	if transform.N1 != 0 {
		_, rotN1, rotN2 := commonlintrans.LinearTransformation(transform).BSGSIndex()
		for _, rotation := range rotN1 {
			if rotation != 0 {
				count++
			}
		}
		for _, rotation := range rotN2 {
			if rotation != 0 {
				count++
			}
		}
		return C.int(count)
	}
	for diagIdx := range transform.Vec {
		if diagIdx != 0 {
			count++
		}
	}
	return C.int(count)
}

//export PlanLinearTransformRotationKeys
func PlanLinearTransformRotationKeys(
	diagIdxsC *C.int, diagIdxsLen C.int,
	level C.int,
	bsgsRatio C.float,
) (*C.int, C.ulong) {
	diagIdxs := CArrayToSlice(diagIdxsC, diagIdxsLen, convertCIntToInt)
	ltparams := lintrans.Parameters{
		DiagonalsIndexList:        diagIdxs,
		LevelQ:                    int(level),
		LevelP:                    scheme.Params.MaxLevelP(),
		Scale:                     rlwe.NewScale(scheme.Params.Q()[int(level)]),
		LogDimensions:             ring.Dimensions{Rows: 0, Cols: scheme.Params.LogMaxSlots()},
		LogBabyStepGiantStepRatio: int(math.Log(float64(bsgsRatio))),
	}
	transform := newLinearTransformationShell(ltparams, 0)
	galEls := transform.GaloisElements(scheme.Params)
	arrPtr, length := SliceToCArray(galEls, convertULongtoInt)
	return arrPtr, length
}

//export PlanLinearTransformRotationKeyRequests
func PlanLinearTransformRotationKeyRequests(
	diagIdxsC *C.int, diagIdxsLen C.int,
	level C.int,
	bsgsRatio C.float,
) (*C.int, C.ulong) {
	diagIdxs := CArrayToSlice(diagIdxsC, diagIdxsLen, convertCIntToInt)
	ltparams := lintrans.Parameters{
		DiagonalsIndexList:        diagIdxs,
		LevelQ:                    int(level),
		LevelP:                    scheme.Params.MaxLevelP(),
		Scale:                     rlwe.NewScale(scheme.Params.Q()[int(level)]),
		LogDimensions:             ring.Dimensions{Rows: 0, Cols: scheme.Params.LogMaxSlots()},
		LogBabyStepGiantStepRatio: int(math.Log(float64(bsgsRatio))),
	}
	transform := newLinearTransformationShell(ltparams, 0)
	galEls := transform.GaloisElements(scheme.Params)
	flat := make([]int, 0, len(galEls)*2)
	for _, galEl := range galEls {
		flat = append(flat, int(galEl), int(level))
	}
	arrPtr, length := SliceToCArray(flat, convertIntToCInt)
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
	changed := ensureLinearTransformEvalKeysMutable()
	key := uint64(galEl)
	if rotKey, exists := scheme.EvalKeys.GaloisKeys[key]; exists && rotKey != nil {
		if changed {
			bumpLinearTransformEvalKeysVersion()
		}
		return
	}
	rotKey := scheme.KeyGen.GenGaloisKeyNew(key, scheme.SecretKey)
	scheme.EvalKeys.GaloisKeys[key] = rotKey
	changed = true
	if changed {
		bumpLinearTransformEvalKeysVersion()
	}
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
	ensureLinearTransformEvalKeysMutable()
	scheme.EvalKeys.GaloisKeys[uint64(galEl)] = &rotKey
	bumpLinearTransformEvalKeysVersion()
}

//export PredecodeRotationKey
func PredecodeRotationKey(
	dataPtr *C.uchar, lenData C.ulong,
	galEl C.ulong,
) {
	rotKeySerial := CArrayToByteSlice(unsafe.Pointer(dataPtr), uint64(lenData))

	var rotKey rlwe.GaloisKey
	if err := rotKey.UnmarshalBinary(rotKeySerial); err != nil {
		panic(err)
	}

	ltPredecodeMu.Lock()
	ltPredecodedRotationKeys[uint64(galEl)] = &rotKey
	ltPredecodeMu.Unlock()
}

//export InstallPredecodedRotationKey
func InstallPredecodedRotationKey(galEl C.ulong) C.int {
	key := uint64(galEl)
	ltPredecodeMu.Lock()
	rotKey, ok := ltPredecodedRotationKeys[key]
	if ok {
		delete(ltPredecodedRotationKeys, key)
	}
	ltPredecodeMu.Unlock()

	if !ok || rotKey == nil {
		return C.int(0)
	}
	ensureLinearTransformEvalKeysMutable()
	scheme.EvalKeys.GaloisKeys[key] = rotKey
	bumpLinearTransformEvalKeysVersion()
	return C.int(1)
}

//export RemovePredecodedRotationKeys
func RemovePredecodedRotationKeys() {
	ltPredecodeMu.Lock()
	ltPredecodedRotationKeys = make(map[uint64]*rlwe.GaloisKey)
	ltPredecodeMu.Unlock()
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

//export PredecodePlaintextDiagonalsBatch
func PredecodePlaintextDiagonalsBatch(
	dataPtr *C.uchar, lenData C.ulong,
	offsetsPtr *C.ulonglong, offsetsLen C.int,
	lengthsPtr *C.ulonglong, lengthsLen C.int,
	diagIdxsPtr *C.int, diagIdxsLen C.int,
	transformID C.int,
) {
	if int(offsetsLen) != int(lengthsLen) || int(offsetsLen) != int(diagIdxsLen) {
		panic("PredecodePlaintextDiagonalsBatch received mismatched batch array lengths")
	}

	payload := CArrayToByteSlice(unsafe.Pointer(dataPtr), uint64(lenData))
	offsets := unsafe.Slice(offsetsPtr, int(offsetsLen))
	lengths := unsafe.Slice(lengthsPtr, int(lengthsLen))
	diagIdxs := unsafe.Slice(diagIdxsPtr, int(diagIdxsLen))

	decoded := make(map[int]ringqp.Poly, int(diagIdxsLen))
	for i := range diagIdxs {
		start := uint64(offsets[i])
		end := start + uint64(lengths[i])
		if end > uint64(len(payload)) {
			panic("PredecodePlaintextDiagonalsBatch slice exceeds payload bounds")
		}

		var poly ringqp.Poly
		if err := poly.UnmarshalBinary(payload[start:end]); err != nil {
			panic(err)
		}
		decoded[int(diagIdxs[i])] = poly
	}

	ltPredecodeMu.Lock()
	existing := ltPredecodedPlaintextVecs[int(transformID)]
	if existing == nil {
		existing = make(map[int]ringqp.Poly, len(decoded))
		ltPredecodedPlaintextVecs[int(transformID)] = existing
	}
	for diagIdx, poly := range decoded {
		existing[diagIdx] = poly
	}
	ltPredecodeMu.Unlock()
}

//export InstallPredecodedPlaintextDiagonals
func InstallPredecodedPlaintextDiagonals(transformID C.int) C.int {
	id := int(transformID)
	ltPredecodeMu.Lock()
	decoded, ok := ltPredecodedPlaintextVecs[id]
	if ok {
		delete(ltPredecodedPlaintextVecs, id)
	}
	ltPredecodeMu.Unlock()

	if !ok {
		return C.int(0)
	}
	transform := RetrieveLinearTransform(id)
	count := 0
	for diagIdx, poly := range decoded {
		transform.Vec[diagIdx] = poly
		count++
	}
	return C.int(count)
}

//export RemovePredecodedPlaintextDiagonals
func RemovePredecodedPlaintextDiagonals(transformID C.int) {
	deletePredecodedPlaintextDiagonals(int(transformID))
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
	scheme.EvalKeys = rlwe.NewMemEvaluationKeySet(scheme.RelinKey)
	bumpLinearTransformEvalKeysVersion()
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

//export PlanLinearTransformsUnifiedRotationKeys
func PlanLinearTransformsUnifiedRotationKeys(
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
	if !unifiedNoBSGSEnabled() {
		applyOptimalUnifiedBSGSRatio(params)
	}
	transforms := newUnifiedLoadTransformations(params)
	keys := make(map[int]struct{})
	for _, transform := range transforms {
		for _, galEl := range transform.GaloisElements(scheme.Params) {
			keys[int(galEl)] = struct{}{}
		}
	}
	ordered := make([]int, 0, len(keys))
	for key := range keys {
		ordered = append(ordered, key)
	}
	sort.Ints(ordered)
	return SliceToCArray(ordered, convertIntToCInt)
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
	linEvaluator := ensureCurrentLinearTransformEvaluator()

	if streamingAny {
		outputs, err := evaluateStreamingLinearTransformsWithSharedCacheNew(
			transformIDsSlice,
			transforms,
			ctIn,
			linEvaluator,
		)
		if err != nil {
			panic(err)
		}
		outIDs := make([]int, n)
		pushStarted := time.Now()
		for i, ct := range outputs {
			outIDs[i] = PushCiphertext(ct)
		}
		recordSharedCacheEvalProfile(func(profile *sharedCacheEvalProfile) {
			profile.pushS += secondsSince(pushStarted)
		})
		return SliceToCArray(outIDs, convertIntToCInt)
	}

	outputs := make([]*rlwe.Ciphertext, n)
	for i := range outputs {
		outputs[i] = rlwe.NewCiphertext(*scheme.Params, 1, transforms[i].LevelQ)
	}

	evalStarted := time.Now()
	if err := linEvaluator.EvaluateManyWithSharedCache(ctIn, transforms, outputs); err != nil {
		panic(err)
	}
	recordSharedCacheEvalProfile(func(profile *sharedCacheEvalProfile) {
		profile.giantStepS += secondsSince(evalStarted)
	})

	outIDs := make([]int, n)
	pushStarted := time.Now()
	for i, ct := range outputs {
		outIDs[i] = PushCiphertext(ct)
	}
	recordSharedCacheEvalProfile(func(profile *sharedCacheEvalProfile) {
		profile.pushS += secondsSince(pushStarted)
	})
	return SliceToCArray(outIDs, convertIntToCInt)
}

//export EvaluateLinearTransformSourcesWithSharedCacheAdd
func EvaluateLinearTransformSourcesWithSharedCacheAdd(
	ctxtIDs *C.int,
	numSources C.int,
	transformIDs *C.int,
	targetIDs *C.int,
	groupOffsets *C.int,
	numPartials C.int,
	numTargets C.int,
) (*C.int, C.ulong) {
	sourceCount := int(numSources)
	partialCount := int(numPartials)
	targetCount := int(numTargets)
	if sourceCount <= 0 {
		panic(fmt.Errorf("EvaluateLinearTransformSourcesWithSharedCacheAdd requires at least one source"))
	}
	if partialCount <= 0 {
		panic(fmt.Errorf("EvaluateLinearTransformSourcesWithSharedCacheAdd requires at least one transform"))
	}
	if targetCount <= 0 {
		panic(fmt.Errorf("EvaluateLinearTransformSourcesWithSharedCacheAdd requires at least one target"))
	}

	ctxtIDsSlice := CArrayToSlice(ctxtIDs, numSources, convertCIntToInt)
	transformIDsSlice := CArrayToSlice(transformIDs, numPartials, convertCIntToInt)
	targetIDsSlice := CArrayToSlice(targetIDs, numPartials, convertCIntToInt)
	offsetsSlice := CArrayToSlice(groupOffsets, C.int(sourceCount+1), convertCIntToInt)
	if len(offsetsSlice) != sourceCount+1 {
		panic(fmt.Errorf("groupOffsets length mismatch"))
	}
	if offsetsSlice[0] != 0 || offsetsSlice[sourceCount] != partialCount {
		panic(fmt.Errorf("invalid group offsets: first=%d last=%d partials=%d", offsetsSlice[0], offsetsSlice[sourceCount], partialCount))
	}

	ctIns := make([]*rlwe.Ciphertext, sourceCount)
	transformGroups := make([][]lintrans.LinearTransformation, sourceCount)
	transformIDGroups := make([][]int, sourceCount)
	targetGroups := make([][]int, sourceCount)
	streamingAny := false
	for sourceIndex := 0; sourceIndex < sourceCount; sourceIndex++ {
		start := offsetsSlice[sourceIndex]
		end := offsetsSlice[sourceIndex+1]
		if start < 0 || end < start || end > partialCount {
			panic(fmt.Errorf("invalid group offset range for source %d: [%d,%d)", sourceIndex, start, end))
		}
		ctIns[sourceIndex] = RetrieveCiphertext(ctxtIDsSlice[sourceIndex])
		groupLen := end - start
		transformGroups[sourceIndex] = make([]lintrans.LinearTransformation, groupLen)
		transformIDGroups[sourceIndex] = make([]int, groupLen)
		targetGroups[sourceIndex] = make([]int, groupLen)
		for localIndex := 0; localIndex < groupLen; localIndex++ {
			partialIndex := start + localIndex
			transformID := transformIDsSlice[partialIndex]
			transform := RetrieveLinearTransform(transformID)
			if hasStreamingLTState(transformID) {
				streamingAny = true
			} else {
				validateLinearTransformPlaintextLevels(transformID, transform)
			}
			ensureLinearTransformRotationKeys(transform)
			transformIDGroups[sourceIndex][localIndex] = transformID
			transformGroups[sourceIndex][localIndex] = transform
			targetGroups[sourceIndex][localIndex] = targetIDsSlice[partialIndex]
		}
	}

	linEvaluator := ensureCurrentLinearTransformEvaluator()

	if streamingAny {
		outputs := make([]*rlwe.Ciphertext, targetCount)
		for sourceIndex := 0; sourceIndex < sourceCount; sourceIndex++ {
			partials, err := evaluateStreamingLinearTransformsWithSharedCacheNew(
				transformIDGroups[sourceIndex],
				transformGroups[sourceIndex],
				ctIns[sourceIndex],
				linEvaluator,
			)
			if err != nil {
				panic(err)
			}
			if len(partials) != len(targetGroups[sourceIndex]) {
				panic(fmt.Errorf("streaming target-sum returned %d partials for %d targets", len(partials), len(targetGroups[sourceIndex])))
			}
			accumulateStarted := time.Now()
			for localIndex, partial := range partials {
				targetID := targetGroups[sourceIndex][localIndex]
				if targetID < 0 || targetID >= targetCount {
					panic(fmt.Errorf("target id out of range: target=%d count=%d", targetID, targetCount))
				}
				if outputs[targetID] == nil {
					outputs[targetID] = partial
					continue
				}
				if err = scheme.Evaluator.Add(outputs[targetID], partial, outputs[targetID]); err != nil {
					panic(err)
				}
			}
			recordSharedCacheEvalProfile(func(profile *sharedCacheEvalProfile) {
				profile.streamAccumulateS += secondsSince(accumulateStarted)
			})
		}

		outIDs := make([]int, targetCount)
		pushStarted := time.Now()
		for i, ct := range outputs {
			if ct == nil {
				panic(fmt.Errorf("target-sum reduction produced no output for target %d", i))
			}
			outIDs[i] = PushCiphertext(ct)
		}
		recordSharedCacheEvalProfile(func(profile *sharedCacheEvalProfile) {
			profile.pushS += secondsSince(pushStarted)
		})
		return SliceToCArray(outIDs, convertIntToCInt)
	}

	outputs := make([]*rlwe.Ciphertext, targetCount)
	for i := range outputs {
		outputs[i] = rlwe.NewCiphertext(*scheme.Params, 1, scheme.Params.MaxLevel())
	}
	evalStarted := time.Now()
	if err := linEvaluator.EvaluateManySourcesWithSharedCacheAdd(
		ctIns,
		transformGroups,
		targetGroups,
		outputs,
	); err != nil {
		panic(err)
	}
	recordSharedCacheEvalProfile(func(profile *sharedCacheEvalProfile) {
		profile.giantStepS += secondsSince(evalStarted)
	})

	outIDs := make([]int, targetCount)
	pushStarted := time.Now()
	for i, ct := range outputs {
		outIDs[i] = PushCiphertext(ct)
	}
	recordSharedCacheEvalProfile(func(profile *sharedCacheEvalProfile) {
		profile.pushS += secondsSince(pushStarted)
	})
	return SliceToCArray(outIDs, convertIntToCInt)
}
