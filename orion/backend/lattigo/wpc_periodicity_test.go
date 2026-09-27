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
