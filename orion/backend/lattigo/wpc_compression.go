package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"sync"
	"time"

	lintrans "github.com/realqhc/lattigo/v6/circuits/ckks/lintrans"
	"github.com/realqhc/lattigo/v6/core/rlwe"
	"github.com/realqhc/lattigo/v6/ring"
	"github.com/realqhc/lattigo/v6/ring/ringqp"
)

// The logical metadata accounting below is intentionally explicit and stable.
// The transform header stores schema version, ring degree, and diagonal count.
// Each diagonal stores key, slot/evaluation periods, Q/P levels, and Q/P limb
// counts. This excludes Go allocator/map overhead and is reported separately
// from the compressed Q/P payload bytes.
const (
	wpcCompressedSchemaVersion uint64 = 1
	wpcTransformMetadataBytes  uint64 = 3 * 8
	wpcDiagonalMetadataBytes   uint64 = 7 * 8
)

type wpcCompressedEvaluationPoly struct {
	Q [][]uint64
	P [][]uint64
}

type wpcCompressedDiagonal struct {
	Key              int
	SlotPeriod       int
	EvaluationPeriod int
	LevelQ           int
	LevelP           int
	Representatives  wpcCompressedEvaluationPoly
}

type wpcCompressedTransformState struct {
	mu sync.Mutex

	RingDegree int
	Diagonals  map[int]wpcCompressedDiagonal

	FullPayloadBytes       uint64
	CompressedPayloadBytes uint64
	MetadataBytes          uint64
	MinSlotPeriod          int
	MaxSlotPeriod          int

	Materialized              bool
	LastDecompressNanoseconds uint64
	LastEvaluateNanoseconds   uint64
	DecompressionCount        uint64
	EvaluationCount           uint64
	OfflineEncodeCalls        uint64
	OnlineEncodeCalls         uint64
}

var (
	wpcCompressedMu         sync.Mutex
	wpcCompressedTransforms = make(map[int]*wpcCompressedTransformState)
)

func compressWPCEvaluationLimbs(
	coeffs [][]uint64,
	evaluationPeriod int,
) (representatives [][]uint64, fullBytes, compressedBytes uint64, ok bool) {
	representatives = make([][]uint64, len(coeffs))
	for limb := range coeffs {
		fullBytes += uint64(len(coeffs[limb])) * 8
		reps, periodic := evaluationBlockRepresentatives(coeffs[limb], evaluationPeriod)
		if !periodic {
			return nil, 0, 0, false
		}
		representatives[limb] = reps
		compressedBytes += uint64(len(reps)) * 8
	}
	return representatives, fullBytes, compressedBytes, true
}

func compressWPCQPEvaluationPoly(
	poly ringqp.Poly,
	slotPeriod int,
) (compressed wpcCompressedEvaluationPoly, fullBytes, compressedBytes uint64, ok bool) {
	evaluationPeriod := 2 * slotPeriod
	q, qFull, qCompressed, qOK := compressWPCEvaluationLimbs(
		poly.Q.Coeffs,
		evaluationPeriod,
	)
	if !qOK {
		return wpcCompressedEvaluationPoly{}, 0, 0, false
	}
	p, pFull, pCompressed, pOK := compressWPCEvaluationLimbs(
		poly.P.Coeffs,
		evaluationPeriod,
	)
	if !pOK {
		return wpcCompressedEvaluationPoly{}, 0, 0, false
	}
	return wpcCompressedEvaluationPoly{Q: q, P: p},
		qFull + pFull,
		qCompressed + pCompressed,
		true
}

func reconstructWPCEvaluationLimbs(
	representatives [][]uint64,
	ringDegree int,
) ([][]uint64, bool) {
	coeffs := make([][]uint64, len(representatives))
	for limb := range representatives {
		reconstructed, ok := reconstructEvaluationBlocks(
			representatives[limb],
			ringDegree,
		)
		if !ok {
			return nil, false
		}
		coeffs[limb] = reconstructed
	}
	return coeffs, true
}

func newRingQPCoefficients(q, p [][]uint64) ringqp.Poly {
	poly := ringqp.Poly{}
	poly.Q.Coeffs = q
	poly.P.Coeffs = p
	return poly
}

func reconstructWPCDiagonal(
	diagonal wpcCompressedDiagonal,
	ringDegree int,
) (ringqp.Poly, bool) {
	if diagonal.SlotPeriod <= 0 ||
		diagonal.EvaluationPeriod != 2*diagonal.SlotPeriod {
		return ringqp.Poly{}, false
	}
	for _, limb := range diagonal.Representatives.Q {
		if len(limb) != diagonal.EvaluationPeriod {
			return ringqp.Poly{}, false
		}
	}
	for _, limb := range diagonal.Representatives.P {
		if len(limb) != diagonal.EvaluationPeriod {
			return ringqp.Poly{}, false
		}
	}
	q, qOK := reconstructWPCEvaluationLimbs(
		diagonal.Representatives.Q,
		ringDegree,
	)
	if !qOK {
		return ringqp.Poly{}, false
	}
	p, pOK := reconstructWPCEvaluationLimbs(
		diagonal.Representatives.P,
		ringDegree,
	)
	if !pOK {
		return ringqp.Poly{}, false
	}
	poly := newRingQPCoefficients(q, p)
	if poly.LevelQ() != diagonal.LevelQ || poly.LevelP() != diagonal.LevelP {
		return ringqp.Poly{}, false
	}
	return poly, true
}

func registerWPCCompressedTransform(
	transformID int,
	state *wpcCompressedTransformState,
) {
	wpcCompressedMu.Lock()
	wpcCompressedTransforms[transformID] = state
	wpcCompressedMu.Unlock()
}

func lookupWPCCompressedTransform(transformID int) (*wpcCompressedTransformState, bool) {
	wpcCompressedMu.Lock()
	state, ok := wpcCompressedTransforms[transformID]
	wpcCompressedMu.Unlock()
	return state, ok
}

func deleteWPCCompressedTransform(transformID int) {
	wpcCompressedMu.Lock()
	delete(wpcCompressedTransforms, transformID)
	wpcCompressedMu.Unlock()
}

func clearWPCCompressedTransforms() {
	wpcCompressedMu.Lock()
	wpcCompressedTransforms = make(map[int]*wpcCompressedTransformState)
	wpcCompressedMu.Unlock()
}

func materializeWPCTransformLocked(
	transformID int,
	state *wpcCompressedTransformState,
) (int, bool) {
	transform := RetrieveLinearTransform(transformID)
	count := 0
	for key, diagonal := range state.Diagonals {
		if diagonal.Key != key {
			return 0, false
		}
		poly, ok := reconstructWPCDiagonal(diagonal, state.RingDegree)
		if !ok {
			return 0, false
		}
		transform.Vec[key] = poly
		count++
	}
	state.Materialized = true
	return count, true
}

func removeWPCMaterializationLocked(
	transformID int,
	state *wpcCompressedTransformState,
) {
	transform := RetrieveLinearTransform(transformID)
	for key := range state.Diagonals {
		transform.Vec[key] = ringqp.Poly{}
	}
	state.Materialized = false
}

// GenerateWPCCompressedLinearTransform performs the offline Encode once,
// retains only one Q/P evaluation period per limb, and discards every full
// plaintext polynomial before returning the transform ID.
//
//export GenerateWPCCompressedLinearTransform
func GenerateWPCCompressedLinearTransform(
	diagIdxsC *C.int, diagIdxsLen C.int,
	diagDataC *C.float, diagDataLen C.int,
	level C.int,
	bsgsRatio C.float,
	slotPeriod C.int,
) C.int {
	if scheme.Params == nil || scheme.Encoder == nil {
		panic("GenerateWPCCompressedLinearTransform requires an initialized scheme and encoder")
	}
	diagIdxs := CArrayToSlice(diagIdxsC, diagIdxsLen, convertCIntToInt)
	slots := scheme.Params.MaxSlots()
	period := int(slotPeriod)
	if period <= 0 || period >= slots || slots%period != 0 {
		panic("GenerateWPCCompressedLinearTransform received an invalid proper slot period")
	}
	if int(diagDataLen) != len(diagIdxs)*slots {
		panic("GenerateWPCCompressedLinearTransform received mismatched diagonal data")
	}

	ltparams := lintrans.Parameters{
		DiagonalsIndexList:        diagIdxs,
		LevelQ:                    int(level),
		LevelP:                    scheme.Params.MaxLevelP(),
		Scale:                     rlwe.NewScale(scheme.Params.Q()[int(level)]),
		LogDimensions:             ring.Dimensions{Rows: 0, Cols: scheme.Params.LogMaxSlots()},
		LogBabyStepGiantStepRatio: int(math.Log(float64(bsgsRatio))),
	}
	transform := lintrans.NewTransformation(scheme.Params, ltparams)
	diagonals := buildFloatDiagonalsFromC(
		diagIdxs,
		diagDataC,
		diagDataLen,
		slots,
	)
	if err := lintrans.Encode(scheme.Encoder, diagonals, transform); err != nil {
		panic(err)
	}

	state := &wpcCompressedTransformState{
		RingDegree:         scheme.Params.N(),
		Diagonals:          make(map[int]wpcCompressedDiagonal, len(transform.Vec)),
		MinSlotPeriod:      period,
		MaxSlotPeriod:      period,
		MetadataBytes:      wpcTransformMetadataBytes,
		OfflineEncodeCalls: 1,
		OnlineEncodeCalls:  0,
	}
	for key, poly := range transform.Vec {
		compressed, fullBytes, compressedBytes, ok := compressWPCQPEvaluationPoly(
			poly,
			period,
		)
		if !ok {
			panic("encoded WPC diagonal did not have the required Q/P evaluation period")
		}
		state.Diagonals[key] = wpcCompressedDiagonal{
			Key:              key,
			SlotPeriod:       period,
			EvaluationPeriod: 2 * period,
			LevelQ:           poly.LevelQ(),
			LevelP:           poly.LevelP(),
			Representatives:  compressed,
		}
		state.FullPayloadBytes += fullBytes
		state.CompressedPayloadBytes += compressedBytes
		state.MetadataBytes += wpcDiagonalMetadataBytes
		transform.Vec[key] = ringqp.Poly{}
	}

	transformID := AddLinearTransform(transform)
	registerWPCCompressedTransform(transformID, state)
	return C.int(transformID)
}

// DecompressWPCLinearTransform reconstructs full Q/P plaintexts directly from
// the stored representatives. It never invokes a Lattigo encoder.
//
//export DecompressWPCLinearTransform
func DecompressWPCLinearTransform(transformID C.int) C.int {
	state, ok := lookupWPCCompressedTransform(int(transformID))
	if !ok {
		return C.int(0)
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	started := time.Now()
	count, reconstructed := materializeWPCTransformLocked(int(transformID), state)
	state.LastDecompressNanoseconds = uint64(time.Since(started).Nanoseconds())
	state.DecompressionCount++
	if !reconstructed {
		removeWPCMaterializationLocked(int(transformID), state)
		return C.int(0)
	}
	return C.int(count)
}

// VerifyWPCDecompressedLinearTransformExact compares every reconstructed Q/P
// coefficient with a separately generated ordinary transform.
//
//export VerifyWPCDecompressedLinearTransformExact
func VerifyWPCDecompressedLinearTransformExact(
	fullTransformID C.int,
	compressedTransformID C.int,
) C.int {
	state, ok := lookupWPCCompressedTransform(int(compressedTransformID))
	if !ok {
		return C.int(0)
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	if !state.Materialized {
		return C.int(0)
	}
	full := RetrieveLinearTransform(int(fullTransformID))
	decompressed := RetrieveLinearTransform(int(compressedTransformID))
	if len(full.Vec) != len(decompressed.Vec) {
		return C.int(0)
	}
	for key, fullPoly := range full.Vec {
		decompressedPoly, exists := decompressed.Vec[key]
		if !exists || !fullPoly.Equal(&decompressedPoly) {
			return C.int(0)
		}
	}
	return C.int(1)
}

//export RemoveWPCDecompressedLinearTransform
func RemoveWPCDecompressedLinearTransform(transformID C.int) {
	state, ok := lookupWPCCompressedTransform(int(transformID))
	if !ok {
		return
	}
	state.mu.Lock()
	removeWPCMaterializationLocked(int(transformID), state)
	state.mu.Unlock()
}

// EvaluateWPCCompressedLinearTransform is the online path. It materializes the
// full plaintext by coefficient copies, evaluates it, and releases the full
// materialization before returning.
//
//export EvaluateWPCCompressedLinearTransform
func EvaluateWPCCompressedLinearTransform(transformID, ctxtID C.int) C.int {
	state, ok := lookupWPCCompressedTransform(int(transformID))
	if !ok {
		panic("EvaluateWPCCompressedLinearTransform received an unknown transform")
	}
	state.mu.Lock()
	defer func() {
		if state.Materialized {
			removeWPCMaterializationLocked(int(transformID), state)
		}
		state.mu.Unlock()
	}()
	decompressStarted := time.Now()
	_, reconstructed := materializeWPCTransformLocked(int(transformID), state)
	state.LastDecompressNanoseconds = uint64(time.Since(decompressStarted).Nanoseconds())
	state.DecompressionCount++
	if !reconstructed {
		panic("EvaluateWPCCompressedLinearTransform failed to reconstruct Q/P plaintexts")
	}

	evaluateStarted := time.Now()
	outputID := EvaluateLinearTransform(transformID, ctxtID)
	state.LastEvaluateNanoseconds = uint64(time.Since(evaluateStarted).Nanoseconds())
	state.EvaluationCount++
	return outputID
}

// GetWPCCompressedLinearTransformStats returns, in order:
// diagonal count, full payload bytes, compressed payload bytes, metadata bytes,
// stored bytes, min/max slot period, last decompression/evaluation nanoseconds,
// offline and online Encode calls, decompression/evaluation counts, and
// currently materialized full bytes.
//
//export GetWPCCompressedLinearTransformStats
func GetWPCCompressedLinearTransformStats(
	transformID C.int,
) (*C.ulonglong, C.ulonglong) {
	state, ok := lookupWPCCompressedTransform(int(transformID))
	if !ok {
		return nil, 0
	}
	state.mu.Lock()
	materializedBytes := uint64(0)
	if state.Materialized {
		materializedBytes = state.FullPayloadBytes
	}
	values := []uint64{
		uint64(len(state.Diagonals)),
		state.FullPayloadBytes,
		state.CompressedPayloadBytes,
		state.MetadataBytes,
		state.CompressedPayloadBytes + state.MetadataBytes,
		uint64(state.MinSlotPeriod),
		uint64(state.MaxSlotPeriod),
		state.LastDecompressNanoseconds,
		state.LastEvaluateNanoseconds,
		state.OfflineEncodeCalls,
		state.OnlineEncodeCalls,
		state.DecompressionCount,
		state.EvaluationCount,
		materializedBytes,
		wpcCompressedSchemaVersion,
	}
	state.mu.Unlock()
	result, length := SliceToCArray(values, convertUint64ToCULonglong)
	return result, C.ulonglong(length)
}
