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

func TestWPCOnlineRecipeExpansionMatchesOrdinaryEncode(t *testing.T) {
	params := wpcTestParameters(t)
	indices := []int{0, 1, 7, 511}
	data := []float32{}
	for _, key := range indices {
		for i := 0; i < params.MaxSlots(); i++ {
			data = append(data, float32(key+i%8)/100)
		}
	}
	ltparams := lintrans.Parameters{DiagonalsIndexList: indices, LevelQ: params.MaxLevel(), LevelP: params.MaxLevelP(),
		Scale: rlwe.NewScale(params.Q()[params.MaxLevel()]), LogDimensions: ring.Dimensions{Cols: params.LogMaxSlots()},
		LogBabyStepGiantStepRatio: 0}
	state := makeWPCOnlineRecipe(indices, data, 8, ltparams)
	require.Equal(t, uint64(len(indices)*8*4), state.RecipeBytes)
	require.Zero(t, state.EncodeCalls)
	data[0] = 999 // retained recipe owns its payload
	require.NotEqual(t, float32(999), state.Periods[0][0])
	full := lintrans.NewTransformation(params, ltparams)
	online := lintrans.NewTransformation(params, ltparams)
	expanded := state.expand()
	ordinary := make(lintrans.Diagonals[float64])
	for _, key := range indices {
		values := make([]float64, params.MaxSlots())
		for i := range values {
			values[i] = float64(float32(key+i%8) / 100)
		}
		ordinary[key] = values
	}
	require.NoError(t, lintrans.Encode(ckks.NewEncoder(params), ordinary, full))
	require.NoError(t, lintrans.Encode(ckks.NewEncoder(params), expanded, online))
	for key, poly := range full.Vec {
		other := online.Vec[key]
		require.True(t, poly.Equal(&other))
	}
	shell := newLinearTransformationShell(ltparams, 0)
	require.Equal(t, full.N1, shell.N1)
	require.ElementsMatch(t, full.GaloisElements(params), shell.GaloisElements(params))
	for _, poly := range shell.Vec {
		require.Empty(t, poly.Q.Coeffs)
		require.Empty(t, poly.P.Coeffs)
	}
}

func TestWPCOnlineRecipeRejectsInvalidData(t *testing.T) {
	params := wpcTestParameters(t)
	ltparams := lintrans.Parameters{DiagonalsIndexList: []int{0}, LevelQ: 1, LevelP: 0,
		LogDimensions: ring.Dimensions{Cols: params.LogMaxSlots()}}
	data := make([]float32, params.MaxSlots())
	require.Panics(t, func() { makeWPCOnlineRecipe([]int{0}, data, params.MaxSlots(), ltparams) })
	require.Panics(t, func() { makeWPCOnlineRecipe([]int{0}, data[:10], 8, ltparams) })
	data[9] = 1
	require.Panics(t, func() { makeWPCOnlineRecipe([]int{0}, data, 8, ltparams) })
	data[9] = 0
	require.Panics(t, func() { makeWPCOnlineRecipe([]int{0, 0}, append(data, data...), 8, ltparams) })
	data[0] = float32(math.NaN())
	require.Panics(t, func() { makeWPCOnlineRecipe([]int{0}, data, 8, ltparams) })
}

func TestWPCOnlineEncodeReleasesOnEvaluatorPanic(t *testing.T) {
	params := wpcTestParameters(t)
	previous := scheme
	scheme.Params, scheme.Encoder = &params, ckks.NewEncoder(params)
	defer func() { scheme = previous; clearWPCOnlineRecipes() }()
	clearWPCOnlineRecipes()
	ltparams := lintrans.Parameters{DiagonalsIndexList: []int{0}, LevelQ: params.MaxLevel(), LevelP: params.MaxLevelP(),
		Scale: rlwe.NewScale(params.Q()[params.MaxLevel()]), LogDimensions: ring.Dimensions{Cols: params.LogMaxSlots()},
		LogBabyStepGiantStepRatio: 0}
	state := makeWPCOnlineRecipe([]int{0}, make([]float32, params.MaxSlots()), 8, ltparams)
	id := AddLinearTransform(newLinearTransformationShell(ltparams, 0))
	defer ltHeap.Delete(id)
	wpcOnlineRecipes[id] = state
	require.Panics(t, func() { evaluateWPCOnlineLinearTransform(id, -999999) })
	require.Equal(t, uint64(1), state.EncodeCalls) // Encode succeeded, invalid input failed evaluation
	require.Zero(t, wpcOnlineCurrentBytes)
	require.Zero(t, wpcOnlineCurrentTransforms)
	for _, poly := range RetrieveLinearTransform(id).Vec {
		require.Empty(t, poly.Q.Coeffs)
		require.Empty(t, poly.P.Coeffs)
	}
}

func TestWPCOnlineRegistryAndEncodeCountersClear(t *testing.T) {
	clearWPCOnlineRecipes()
	defer clearWPCOnlineRecipes()
	for i := range wpcBenchmarkEncodeCounts {
		recordWPCBenchmarkEncode(i)
	}
	require.Equal(t, [3]uint64{1, 1, 1}, wpcBenchmarkEncodeCounts)
	wpcOnlineRecipes[123] = &wpcOnlineRecipe{}
	deleteWPCOnlineRecipe(123)
	require.Empty(t, wpcOnlineRecipes)
	clearWPCOnlineRecipes()
	require.Equal(t, [3]uint64{}, wpcBenchmarkEncodeCounts)
}
