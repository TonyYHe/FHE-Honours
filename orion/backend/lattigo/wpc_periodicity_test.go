package main

import (
	"fmt"
	"testing"

	lintrans "github.com/realqhc/lattigo/v6/circuits/ckks/lintrans"
	"github.com/realqhc/lattigo/v6/core/rlwe"
	"github.com/realqhc/lattigo/v6/ring"
	"github.com/realqhc/lattigo/v6/ring/ringqp"
	"github.com/realqhc/lattigo/v6/schemes/ckks"
	"github.com/stretchr/testify/require"
)

func requireExactEvaluationPeriod(t *testing.T, poly ring.Poly, evaluationPeriod int) {
	t.Helper()
	require.NotEmpty(t, poly.Coeffs)
	for limb, coeffs := range poly.Coeffs {
		representatives, periodic := evaluationBlockRepresentatives(coeffs, evaluationPeriod)
		require.Truef(t, periodic, "limb %d is not exactly periodic", limb)
		reconstructed, ok := reconstructEvaluationBlocks(representatives, len(coeffs))
		require.True(t, ok)
		require.Equalf(t, coeffs, reconstructed, "limb %d failed exact reconstruction", limb)
	}
}

func requireExactQPEvaluationPeriod(t *testing.T, poly ringqp.Poly, evaluationPeriod int) {
	t.Helper()
	requireExactEvaluationPeriod(t, poly.Q, evaluationPeriod)
	requireExactEvaluationPeriod(t, poly.P, evaluationPeriod)
}

func periodicValues(slots, period, seed int) []float64 {
	values := make([]float64, slots)
	for i := range values {
		values[i] = float64(seed + i%period + 1)
	}
	return values
}

func wpcTestParameters(t *testing.T) ckks.Parameters {
	t.Helper()
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            10,
		LogQ:            []int{55, 45, 45, 45, 45, 45, 45},
		LogP:            []int{60},
		LogDefaultScale: 45,
		RingType:        ring.Standard,
	})
	require.NoError(t, err)
	return params
}

func wpcTestTransformation(params ckks.Parameters) lintrans.LinearTransformation {
	return lintrans.NewTransformation(params, lintrans.Parameters{
		DiagonalsIndexList:        []int{0},
		LevelQ:                    params.MaxLevel(),
		LevelP:                    params.MaxLevelP(),
		Scale:                     rlwe.NewScale(params.Q()[params.MaxLevel()]),
		LogDimensions:             ring.Dimensions{Rows: 0, Cols: params.LogMaxSlots()},
		LogBabyStepGiantStepRatio: -1,
	})
}

func TestWPCExactPeriodAndQPRoundTrip(t *testing.T) {
	params := wpcTestParameters(t)
	encoder := ckks.NewEncoder(params)
	slots := params.MaxSlots()
	for _, slotPeriod := range []int{1, 2, 8, 32, slots / 4} {
		t.Run(fmt.Sprintf("T=%d", slotPeriod), func(t *testing.T) {
			transform := wpcTestTransformation(params)
			require.NoError(t, lintrans.Encode(
				encoder,
				lintrans.Diagonals[float64]{0: periodicValues(slots, slotPeriod, 1000*slotPeriod)},
				transform,
			))
			requireExactQPEvaluationPeriod(t, transform.Vec[0], 2*slotPeriod)
		})
	}
}

func TestWPCRejectsNonPeriodicEvaluationPolynomial(t *testing.T) {
	params := wpcTestParameters(t)
	values := make([]float64, params.MaxSlots())
	for i := range values {
		values[i] = float64(i + 1)
	}
	transform := wpcTestTransformation(params)
	require.NoError(t, lintrans.Encode(
		ckks.NewEncoder(params),
		lintrans.Diagonals[float64]{0: values},
		transform,
	))
	for _, coeffs := range transform.Vec[0].Q.Coeffs {
		_, periodic := evaluationBlockRepresentatives(coeffs, 16)
		require.False(t, periodic)
	}
	for _, coeffs := range transform.Vec[0].P.Coeffs {
		_, periodic := evaluationBlockRepresentatives(coeffs, 16)
		require.False(t, periodic)
	}
}

func TestWPCEncodedCandidateVerifier(t *testing.T) {
	params := wpcTestParameters(t)
	for _, slotPeriod := range []int{1, 2, 8, 32, params.MaxSlots() / 4} {
		values := periodicValues(params.MaxSlots(), slotPeriod, 1000*slotPeriod)
		require.Equal(
			t,
			wpcVerificationPassed,
			verifyWPCEncodedReal(params, values, params.MaxLevel(), slotPeriod),
		)
	}

	nonPeriodic := periodicValues(params.MaxSlots(), 8, 100)
	nonPeriodic[len(nonPeriodic)-1] += 1
	require.Equal(
		t,
		wpcVerificationSourceNotPeriodic,
		verifyWPCEncodedReal(params, nonPeriodic, params.MaxLevel(), 8),
	)
}

func TestWPCCompressedQPPayloadRoundTrip(t *testing.T) {
	params := wpcTestParameters(t)
	slotPeriod := 8
	transform := wpcTestTransformation(params)
	require.NoError(t, lintrans.Encode(
		ckks.NewEncoder(params),
		lintrans.Diagonals[float64]{
			0: periodicValues(params.MaxSlots(), slotPeriod, 912),
		},
		transform,
	))

	original := transform.Vec[0]
	compressed, fullBytes, compressedBytes, ok := compressWPCQPEvaluationPoly(
		original,
		slotPeriod,
	)
	require.True(t, ok)
	require.Equal(t, uint64((params.MaxLevel()+1+params.MaxLevelP()+1)*params.N()*8), fullBytes)
	require.Equal(t, uint64((params.MaxLevel()+1+params.MaxLevelP()+1)*2*slotPeriod*8), compressedBytes)
	require.Equal(t, uint64(params.MaxSlots()/slotPeriod), fullBytes/compressedBytes)

	reconstructed, ok := reconstructWPCDiagonal(
		wpcCompressedDiagonal{
			Key:              0,
			SlotPeriod:       slotPeriod,
			EvaluationPeriod: 2 * slotPeriod,
			LevelQ:           original.LevelQ(),
			LevelP:           original.LevelP(),
			Representatives:  compressed,
		},
		params.N(),
	)
	require.True(t, ok)
	require.True(t, original.Equal(&reconstructed))

	_, ok = reconstructWPCDiagonal(
		wpcCompressedDiagonal{
			Key:              0,
			SlotPeriod:       slotPeriod,
			EvaluationPeriod: 2*slotPeriod + 1,
			LevelQ:           original.LevelQ(),
			LevelP:           original.LevelP(),
			Representatives:  compressed,
		},
		params.N(),
	)
	require.False(t, ok)
}

func TestWPCCompressionRejectsWrongEvaluationPeriod(t *testing.T) {
	params := wpcTestParameters(t)
	transform := wpcTestTransformation(params)
	require.NoError(t, lintrans.Encode(
		ckks.NewEncoder(params),
		lintrans.Diagonals[float64]{
			0: periodicValues(params.MaxSlots(), 8, 321),
		},
		transform,
	))

	_, _, _, ok := compressWPCQPEvaluationPoly(transform.Vec[0], 4)
	require.False(t, ok)
}

func TestWPCGlobalAccountingTracksSequentialSingleTransformPeak(t *testing.T) {
	clearWPCCompressedTransforms()
	defer clearWPCCompressedTransforms()

	first := &wpcCompressedTransformState{
		FullPayloadBytes:       100,
		CompressedPayloadBytes: 10,
		MetadataBytes:          5,
		OfflineEncodeCalls:     1,
	}
	second := &wpcCompressedTransformState{
		FullPayloadBytes:       240,
		CompressedPayloadBytes: 24,
		MetadataBytes:          7,
		OfflineEncodeCalls:     1,
	}
	registerWPCCompressedTransform(1001, first)
	registerWPCCompressedTransform(1002, second)

	beginWPCMaterialization(first)
	endWPCMaterialization(first)
	ResetWPCCompressedGlobalMaterializationPeak()
	beginWPCMaterialization(second)
	endWPCMaterialization(second)

	wpcCompressedGlobalMu.Lock()
	snapshot := wpcCompressedGlobal
	wpcCompressedGlobalMu.Unlock()
	require.Equal(t, uint64(2), snapshot.RegisteredTransformCount)
	require.Equal(t, uint64(340), snapshot.AggregateFullPayloadBytes)
	require.Equal(t, uint64(34), snapshot.AggregateCompressedBytes)
	require.Equal(t, uint64(12), snapshot.AggregateMetadataBytes)
	require.Equal(t, uint64(0), snapshot.CurrentMaterializedBytes)
	require.Equal(t, uint64(240), snapshot.PeakMaterializedBytes)
	require.Equal(t, uint64(0), snapshot.CurrentMaterializedTransforms)
	require.Equal(t, uint64(1), snapshot.PeakMaterializedTransforms)
	require.Equal(t, uint64(2), snapshot.OfflineEncodeCalls)
	require.Equal(t, uint64(0), snapshot.OnlineEncodeCalls)
}
