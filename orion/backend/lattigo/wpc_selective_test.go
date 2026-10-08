package main

import (
	"math"
	"testing"

	lintrans "github.com/realqhc/lattigo/v6/circuits/ckks/lintrans"
	"github.com/realqhc/lattigo/v6/core/rlwe"
	"github.com/realqhc/lattigo/v6/ring"
	"github.com/realqhc/lattigo/v6/schemes/ckks"
	"github.com/stretchr/testify/require"
)

func TestWPCSelectiveMixedExactPreservesFullBSGS(t *testing.T) {
	params := wpcTestParameters(t)
	previous := scheme
	scheme.Params, scheme.Encoder = &params, ckks.NewEncoder(params)
	defer func() { scheme = previous; clearWPCSelective() }()
	slots := params.MaxSlots()
	indices := []int{0, 1, 7, -1}
	data := make([]float32, slots*len(indices))
	for k := range indices {
		for i := 0; i < slots; i++ {
			if k == 0 {
				data[k*slots+i] = .25
			}
			if k == 1 {
				data[k*slots+i] = float32(i%8) / 100
			}
			if k == 2 {
				data[k*slots+i] = float32(i) / 100
			}
			// -1 is all zero, intentionally FALLBACK, not eligible.
		}
	}
	for _, bsgs := range []int{-1, 0, 1, 2} {
		for _, levelP := range []int{-1, params.MaxLevelP()} {
			ltparams := lintrans.Parameters{DiagonalsIndexList: indices, LevelQ: 2, LevelP: levelP,
				Scale: rlwe.NewScale(params.Q()[2]), LogDimensions: ring.Dimensions{Cols: params.LogMaxSlots()},
				LogBabyStepGiantStepRatio: bsgs}
			full := lintrans.NewTransformation(params, ltparams)
			values := make(lintrans.Diagonals[float64])
			for k, key := range indices {
				row := make([]float64, slots)
				for i := range row {
					row[i] = float64(data[k*slots+i])
				}
				values[key] = row
			}
			require.NoError(t, lintrans.Encode(scheme.Encoder, values, full))
			for _, hybrid := range []bool{false, true} {
				shell, state, err := makeWPCSelective(indices, data, ltparams, hybrid)
				require.NoError(t, err)
				require.Equal(t, full.N1, shell.N1)
				require.ElementsMatch(t, full.GaloisElements(params), shell.GaloisElements(params))
				require.Equal(t, slots, state.Periods[7])
				require.Equal(t, slots, state.Periods[slots-1])
				id := AddLinearTransform(shell)
				wpcSelectiveTransforms[id] = state
				for attempt := 0; attempt < 2; attempt++ {
					require.NoError(t, materializeWPCSelective(id, indices, data))
					actual := RetrieveLinearTransform(id)
					for key, poly := range full.Vec {
						other := actual.Vec[key]
						require.True(t, poly.Equal(&other))
					}
					require.Error(t, materializeWPCSelective(id, indices, data))
					releaseWPCSelectiveLocked(id, state)
					require.Zero(t, state.CurrentBytes)
					for _, poly := range actual.Vec {
						require.Empty(t, poly.Q.Coeffs)
						require.Empty(t, poly.P.Coeffs)
					}
				}
				if hybrid {
					require.Equal(t, uint64(2), state.OfflineEmbedCalls)
					require.Equal(t, uint64(4), state.OnlineEmbedCalls)
					require.Zero(t, state.OnlineEligibleEmbedCalls)
					require.Zero(t, state.EligibleEncodeNS)
				} else {
					require.Zero(t, state.OfflineEmbedCalls)
					require.Equal(t, uint64(8), state.OnlineEmbedCalls)
					require.Equal(t, uint64(4), state.OnlineEligibleEmbedCalls)
					require.Positive(t, state.EligibleEncodeNS)
				}
				mutated := append([]float32(nil), data...)
				mutated[0] += 1
				require.ErrorContains(t, materializeWPCSelective(id, indices, mutated), "identity")
				require.Zero(t, state.CurrentBytes)
				deleteWPCSelective(id)
				ltHeap.Delete(id)
			}
		}
	}
}

func TestWPCSelectivePayloadValidation(t *testing.T) {
	data := make([]float32, 16)
	_, _, err := selectivePayload([]int{0}, data[:2], 16)
	require.Error(t, err)
	_, _, err = selectivePayload([]int{0, -16}, append(data, data...), 16)
	require.Error(t, err)
	data[0] = float32(math.NaN())
	_, _, err = selectivePayload([]int{0}, data, 16)
	require.Error(t, err)
	clearWPCSelective()
	require.Empty(t, wpcSelectiveTransforms)
}

func TestWPCSelectiveRejectsUnsupportedRingAndSlotDimensions(t *testing.T) {
	params := wpcTestParameters(t)
	previous := scheme
	defer func() { scheme = previous }()
	scheme.Params, scheme.Encoder = &params, ckks.NewEncoder(params)
	ltparams := lintrans.Parameters{LogDimensions: ring.Dimensions{Cols: params.LogMaxSlots() - 1}}
	_, _, err := makeWPCSelective([]int{0}, make([]float32, params.MaxSlots()/2), ltparams, true)
	require.ErrorContains(t, err, "full-slot")
	conjugate, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN: params.LogN(), LogQ: []int{45, 30, 30}, LogP: []int{45}, LogDefaultScale: params.LogDefaultScale(),
		RingType: ring.ConjugateInvariant,
	})
	require.NoError(t, err)
	scheme.Params = &conjugate
	ltparams.LogDimensions.Cols = conjugate.LogMaxSlots()
	_, _, err = makeWPCSelective([]int{0}, make([]float32, conjugate.MaxSlots()), ltparams, true)
	require.ErrorContains(t, err, "standard ring")
}
