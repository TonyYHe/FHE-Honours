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

// evaluationBlockRepresentatives captures the copy map induced by Lattigo's
// bit-reversed NTT storage. A slot period T becomes 2T constant raw blocks,
// each containing N/(2T) equal coefficients.
func evaluationBlockRepresentatives(coeffs []uint64, evaluationPeriod int) ([]uint64, bool) {
	if evaluationPeriod <= 0 || len(coeffs)%evaluationPeriod != 0 {
		return nil, false
	}
	blockSize := len(coeffs) / evaluationPeriod
	representatives := make([]uint64, evaluationPeriod)
	for block := range representatives {
		start := block * blockSize
		representatives[block] = coeffs[start]
		for i := start + 1; i < start+blockSize; i++ {
			if coeffs[i] != representatives[block] {
				return nil, false
			}
		}
	}
	return representatives, true
}

func reconstructEvaluationBlocks(representatives []uint64, ringDegree int) ([]uint64, bool) {
	if len(representatives) == 0 || ringDegree%len(representatives) != 0 {
		return nil, false
	}
	blockSize := ringDegree / len(representatives)
	reconstructed := make([]uint64, ringDegree)
	for block, representative := range representatives {
		start := block * blockSize
		for i := start; i < start+blockSize; i++ {
			reconstructed[i] = representative
		}
	}
	return reconstructed, true
}

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
