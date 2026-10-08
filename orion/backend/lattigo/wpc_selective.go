package main

/*
#include <stdint.h>
*/
import "C"

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"
	"unsafe"

	lintrans "github.com/realqhc/lattigo/v6/circuits/ckks/lintrans"
	"github.com/realqhc/lattigo/v6/core/rlwe"
	"github.com/realqhc/lattigo/v6/ring"
	"github.com/realqhc/lattigo/v6/ring/ringqp"
)

// Selective storage preserves the ORIGINAL transform's N1 and diagonal set.
// Encoding a smaller standalone transform would change BSGS pre-rotations.
// Only eligible encoded periods are retained; fallback values are supplied by
// the existing Python recipe on each materialization, never cached here.
type wpcSelectiveState struct {
	mu                                                                      sync.Mutex
	Parameters                                                              lintrans.Parameters
	Indices                                                                 []int
	Digest                                                                  [32]byte
	Periods                                                                 map[int]int
	Compressed                                                              map[int]wpcCompressedDiagonal
	Hybrid                                                                  bool
	FullBytes, EligibleFullBytes, CompressedBytes, MetadataBytes            uint64
	OfflineEmbedCalls, OnlineEmbedCalls, OnlineEligibleEmbedCalls           uint64
	Materializations, CurrentBytes, PeakBytes                               uint64
	PrepareNS, EncodeNS, EligibleEncodeNS, DecompressNS                     uint64
	TotalPrepareNS, TotalEncodeNS, TotalEligibleEncodeNS, TotalDecompressNS uint64
	LastEmbedNS                                                             map[int]uint64
}

var wpcSelectiveMu sync.Mutex
var wpcSelectiveTransforms = make(map[int]*wpcSelectiveState)

func selectivePayload(indices []int, data []float32, slots int) (map[int][]float32, [32]byte, error) {
	var digest [32]byte
	if slots <= 0 || len(indices) == 0 || len(data) != slots*len(indices) {
		return nil, digest, fmt.Errorf("selective payload dimensions do not match")
	}
	values := make(map[int][]float32, len(indices))
	hash := sha256.New()
	var bytes [4]byte
	for i, key := range indices {
		if key < -slots || key >= slots {
			return nil, digest, fmt.Errorf("selective diagonal index out of range")
		}
		key = normalizeDiagIndex(key, slots)
		if _, exists := values[key]; exists {
			return nil, digest, fmt.Errorf("duplicate normalized selective diagonal")
		}
		binary.LittleEndian.PutUint32(bytes[:], uint32(key))
		_, _ = hash.Write(bytes[:])
		row := data[i*slots : (i+1)*slots]
		for _, value := range row {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, digest, fmt.Errorf("nonfinite selective slot value")
			}
			binary.LittleEndian.PutUint32(bytes[:], math.Float32bits(value))
			_, _ = hash.Write(bytes[:])
		}
		values[key] = row
	}
	copy(digest[:], hash.Sum(nil))
	return values, digest, nil
}

func selectiveSlotPeriod(values []float32) (int, bool) {
	period := len(values)
	for period > 1 {
		half := period / 2
		equal := true
		for i := 0; i < half; i++ {
			if values[i] != values[i+half] {
				equal = false
				break
			}
		}
		if !equal {
			break
		}
		period = half
	}
	for _, value := range values {
		if value != 0 {
			return period, false
		}
	}
	return period, true
}

// One ordinary CKKS Embed, using precisely the pre-rotation of the full BSGS
// transform. Preparation/allocation is outside the returned Encode timer.
func selectiveEmbed(shell lintrans.LinearTransformation, key int, values []float32) (ringqp.Poly, uint64, error) {
	slots := len(values)
	rotation := 0
	if shell.N1 != 0 {
		rotation = -(key / shell.N1 * shell.N1) & (slots - 1)
	}
	embed := make([]float64, slots)
	for i := range embed {
		embed[i] = float64(values[(i+rotation)%slots])
	}
	poly := scheme.Params.RingQP().AtLevel(shell.LevelQ, shell.LevelP).NewPoly()
	metadata := *shell.MetaData
	metadata.Scale = shell.Scale
	started := time.Now()
	err := scheme.Encoder.Embed(embed, &metadata, poly)
	return poly, uint64(time.Since(started).Nanoseconds()), err
}

func makeWPCSelective(indices []int, data []float32, params lintrans.Parameters, hybrid bool) (lintrans.LinearTransformation, *wpcSelectiveState, error) {
	if scheme.Params.RingType() != ring.Standard || params.LogDimensions.Cols != scheme.Params.LogMaxSlots() {
		return lintrans.LinearTransformation{}, nil, fmt.Errorf("selective storage requires a standard ring and full-slot diagonals")
	}
	slots := 1 << params.LogDimensions.Cols
	values, digest, err := selectivePayload(indices, data, slots)
	if err != nil {
		return lintrans.LinearTransformation{}, nil, err
	}
	shell := newLinearTransformationShell(params, 0)
	state := &wpcSelectiveState{Parameters: params, Indices: append([]int(nil), indices...), Digest: digest,
		Periods: make(map[int]int), Compressed: make(map[int]wpcCompressedDiagonal), Hybrid: hybrid,
		MetadataBytes: uint64(8*8 + 32 + 4*len(indices)), LastEmbedNS: make(map[int]uint64)}
	bytesPerDiagonal := uint64(2 * slots * (params.LevelQ + params.LevelP + 2) * 8)
	state.FullBytes = uint64(len(indices)) * bytesPerDiagonal
	for key, row := range values {
		period, zero := selectiveSlotPeriod(row)
		state.Periods[key] = slots // all-zero diagonals remain on the fallback path
		if zero || period == slots {
			continue
		}
		state.Periods[key] = period
		state.EligibleFullBytes += bytesPerDiagonal
		if !hybrid {
			continue
		}
		poly, _, err := selectiveEmbed(shell, key, row)
		if err != nil {
			return shell, nil, err
		}
		compressed, fullBytes, compressedBytes, ok := compressWPCQPEvaluationPoly(poly, period)
		if !ok || fullBytes != bytesPerDiagonal {
			return shell, nil, fmt.Errorf("selective encoded Q/P periodicity failed")
		}
		diagonal := wpcCompressedDiagonal{Key: key, SlotPeriod: period, EvaluationPeriod: 2 * period,
			LevelQ: params.LevelQ, LevelP: params.LevelP, Representatives: compressed}
		reconstructed, ok := reconstructWPCDiagonal(diagonal, 2*slots)
		if !ok || !poly.Equal(&reconstructed) {
			return shell, nil, fmt.Errorf("selective exact reconstruction failed")
		}
		state.Compressed[key] = diagonal
		state.CompressedBytes += compressedBytes
		state.MetadataBytes += wpcDiagonalMetadataBytes
		state.OfflineEmbedCalls++
	}
	return shell, state, nil
}

func releaseWPCSelectiveLocked(id int, state *wpcSelectiveState) {
	shell := RetrieveLinearTransform(id)
	for key := range shell.Vec {
		shell.Vec[key] = ringqp.Poly{}
	}
	state.CurrentBytes = 0
}

func materializeWPCSelective(id int, indices []int, data []float32) error {
	wpcSelectiveMu.Lock()
	state := wpcSelectiveTransforms[id]
	wpcSelectiveMu.Unlock()
	if state == nil {
		return fmt.Errorf("unknown selective transform")
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	if state.CurrentBytes != 0 {
		return fmt.Errorf("selective transform is already materialized")
	}
	started := time.Now()
	values, digest, err := selectivePayload(indices, data, 1<<state.Parameters.LogDimensions.Cols)
	if err != nil {
		return err
	}
	if digest != state.Digest {
		return fmt.Errorf("selective payload identity changed")
	}
	shell := RetrieveLinearTransform(id)
	state.EncodeNS, state.EligibleEncodeNS, state.DecompressNS = 0, 0, 0
	state.LastEmbedNS = make(map[int]uint64)
	keys := make([]int, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Ints(keys)
	completed := false
	defer func() {
		if !completed {
			releaseWPCSelectiveLocked(id, state)
		}
	}()
	for _, key := range keys {
		if diagonal, ok := state.Compressed[key]; ok {
			copyStarted := time.Now()
			poly, ok := reconstructWPCDiagonal(diagonal, scheme.Params.N())
			state.DecompressNS += uint64(time.Since(copyStarted).Nanoseconds())
			if !ok {
				return fmt.Errorf("selective copy-map reconstruction failed")
			}
			shell.Vec[key] = poly
		} else {
			poly, encodeNS, err := selectiveEmbed(shell, key, values[key])
			if err != nil {
				return err
			}
			shell.Vec[key] = poly
			state.EncodeNS += encodeNS
			state.LastEmbedNS[key] = encodeNS
			state.OnlineEmbedCalls++
			if state.Periods[key] < 1<<state.Parameters.LogDimensions.Cols {
				state.EligibleEncodeNS += encodeNS
				state.OnlineEligibleEmbedCalls++
			}
		}
	}
	wallNS := uint64(time.Since(started).Nanoseconds())
	if wallNS < state.EncodeNS+state.DecompressNS {
		return fmt.Errorf("selective timing accounting failed")
	}
	state.PrepareNS = wallNS - state.EncodeNS - state.DecompressNS
	state.TotalPrepareNS += state.PrepareNS
	state.TotalEncodeNS += state.EncodeNS
	state.TotalEligibleEncodeNS += state.EligibleEncodeNS
	state.TotalDecompressNS += state.DecompressNS
	state.Materializations++
	state.CurrentBytes = state.FullBytes
	if state.CurrentBytes > state.PeakBytes {
		state.PeakBytes = state.CurrentBytes
	}
	completed = true
	return nil
}

// New APIs return -1 for rejected input instead of panicking across ctypes.
//
//export GenerateWPCSelectiveLinearTransform
func GenerateWPCSelectiveLinearTransform(indicesC *C.int, indicesLen C.int, dataC *C.float, dataLen C.int, level C.int, ratio C.float, hybrid C.int) C.int {
	if scheme.Params == nil || scheme.Encoder == nil || int(level) < 0 || int(level) > scheme.Params.MaxLevel() ||
		indicesC == nil || dataC == nil || indicesLen <= 0 || dataLen <= 0 ||
		float64(ratio) <= 0 || math.IsNaN(float64(ratio)) || math.IsInf(float64(ratio), 0) || (hybrid != 0 && hybrid != 1) {
		return -1
	}
	indices := CArrayToSlice(indicesC, indicesLen, convertCIntToInt)
	raw := unsafe.Slice(dataC, int(dataLen))
	data := make([]float32, len(raw))
	for i := range raw {
		data[i] = float32(raw[i])
	}
	params := lintrans.Parameters{DiagonalsIndexList: indices, LevelQ: int(level), LevelP: scheme.Params.MaxLevelP(),
		Scale: rlwe.NewScale(scheme.Params.Q()[int(level)]), LogDimensions: ring.Dimensions{Cols: scheme.Params.LogMaxSlots()},
		LogBabyStepGiantStepRatio: int(math.Log(float64(ratio)))}
	shell, state, err := makeWPCSelective(indices, data, params, hybrid == 1)
	if err != nil {
		return -1
	}
	id := AddLinearTransform(shell)
	wpcSelectiveMu.Lock()
	wpcSelectiveTransforms[id] = state
	wpcSelectiveMu.Unlock()
	return C.int(id)
}

//export MaterializeWPCSelectiveLinearTransform
func MaterializeWPCSelectiveLinearTransform(id C.int, indicesC *C.int, indicesLen C.int, dataC *C.float, dataLen C.int) C.int {
	if indicesC == nil || dataC == nil || indicesLen <= 0 || dataLen <= 0 {
		return -1
	}
	indices := CArrayToSlice(indicesC, indicesLen, convertCIntToInt)
	raw := unsafe.Slice(dataC, int(dataLen))
	data := make([]float32, len(raw))
	for i := range raw {
		data[i] = float32(raw[i])
	}
	if err := materializeWPCSelective(int(id), indices, data); err != nil {
		return -1
	}
	return 1
}

//export ReleaseWPCSelectiveLinearTransform
func ReleaseWPCSelectiveLinearTransform(id C.int) {
	wpcSelectiveMu.Lock()
	state := wpcSelectiveTransforms[int(id)]
	wpcSelectiveMu.Unlock()
	if state != nil {
		state.mu.Lock()
		releaseWPCSelectiveLocked(int(id), state)
		state.mu.Unlock()
	}
}

//export VerifyWPCSelectiveLinearTransformExact
func VerifyWPCSelectiveLinearTransformExact(referenceID, id C.int) C.int {
	wpcSelectiveMu.Lock()
	state := wpcSelectiveTransforms[int(id)]
	wpcSelectiveMu.Unlock()
	if state == nil {
		return 0
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	if state.CurrentBytes == 0 {
		return 0
	}
	full, selective := RetrieveLinearTransform(int(referenceID)), RetrieveLinearTransform(int(id))
	if full.N1 != selective.N1 || len(full.Vec) != len(selective.Vec) {
		return 0
	}
	for key, poly := range full.Vec {
		other, ok := selective.Vec[key]
		if !ok || !poly.Equal(&other) {
			return 0
		}
	}
	return 1
}

//export GetWPCSelectiveLinearTransformStats
func GetWPCSelectiveLinearTransformStats(id C.int) (*C.ulonglong, C.ulonglong) {
	wpcSelectiveMu.Lock()
	state := wpcSelectiveTransforms[int(id)]
	wpcSelectiveMu.Unlock()
	if state == nil {
		return nil, 0
	}
	state.mu.Lock()
	eligible := uint64(0)
	for _, period := range state.Periods {
		if period < 1<<state.Parameters.LogDimensions.Cols {
			eligible++
		}
	}
	values := []uint64{1, uint64(len(state.Indices)), eligible, uint64(len(state.Compressed)), state.FullBytes,
		state.EligibleFullBytes, state.CompressedBytes, state.MetadataBytes, state.OfflineEmbedCalls,
		state.OnlineEmbedCalls, state.OnlineEligibleEmbedCalls, state.Materializations, state.CurrentBytes,
		state.PeakBytes, state.PrepareNS, state.EncodeNS, state.EligibleEncodeNS, state.DecompressNS,
		state.TotalPrepareNS, state.TotalEncodeNS, state.TotalEligibleEncodeNS, state.TotalDecompressNS}
	state.mu.Unlock()
	result, length := SliceToCArray(values, convertUint64ToCULonglong)
	return result, C.ulonglong(length)
}

func deleteWPCSelective(id int) {
	wpcSelectiveMu.Lock()
	delete(wpcSelectiveTransforms, id)
	wpcSelectiveMu.Unlock()
}

func clearWPCSelective() {
	wpcSelectiveMu.Lock()
	wpcSelectiveTransforms = make(map[int]*wpcSelectiveState)
	wpcSelectiveMu.Unlock()
}
