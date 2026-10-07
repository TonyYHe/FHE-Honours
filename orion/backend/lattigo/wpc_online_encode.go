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

// A recipe stores one float32 SLOT period, not an encoded Q/P period.
// Key generation needs only the empty transformation shell. No weight Encode
// or full Q/P allocation occurs when registering a recipe.
type wpcOnlineRecipe struct {
	mu                                           sync.Mutex
	Parameters                                   lintrans.Parameters
	Periods                                      map[int][]float32
	Slots                                        int
	FullBytes, RecipeBytes, MetadataBytes        uint64
	PrepareNS, EncodeNS, EvaluateNS, EncodeCalls uint64
}

var wpcOnlineMu sync.Mutex
var wpcOnlineRecipes = make(map[int]*wpcOnlineRecipe)
var wpcOnlineCurrentBytes, wpcOnlinePeakBytes uint64
var wpcOnlineCurrentTransforms, wpcOnlinePeakTransforms uint64
var wpcBenchmarkEncodeMu sync.Mutex
var wpcBenchmarkEncodeCounts [3]uint64 // ordinary, compressed, recipe Encode invocations

func recordWPCBenchmarkEncode(kind int) {
	wpcBenchmarkEncodeMu.Lock()
	wpcBenchmarkEncodeCounts[kind]++
	wpcBenchmarkEncodeMu.Unlock()
}

//export ResetWPCBenchmarkEncodeCounters
func ResetWPCBenchmarkEncodeCounters() {
	wpcBenchmarkEncodeMu.Lock()
	wpcBenchmarkEncodeCounts = [3]uint64{}
	wpcBenchmarkEncodeMu.Unlock()
}

//export GetWPCBenchmarkEncodeCounters
func GetWPCBenchmarkEncodeCounters() (*C.ulonglong, C.ulonglong) {
	wpcBenchmarkEncodeMu.Lock()
	values := append([]uint64(nil), wpcBenchmarkEncodeCounts[:]...)
	wpcBenchmarkEncodeMu.Unlock()
	result, length := SliceToCArray(values, convertUint64ToCULonglong)
	return result, C.ulonglong(length)
}

func makeWPCOnlineRecipe(indices []int, data []float32, period int, params lintrans.Parameters) *wpcOnlineRecipe {
	slots := 1 << params.LogDimensions.Cols
	if period <= 0 || period >= slots || slots%period != 0 || len(indices) == 0 || len(data) != len(indices)*slots {
		panic("invalid WPC online recipe dimensions or period")
	}
	state := &wpcOnlineRecipe{Parameters: params, Slots: slots, Periods: make(map[int][]float32), MetadataBytes: 6 * 8}
	for i, key := range indices {
		if key < 0 || key >= slots {
			panic("WPC recipe requires normalized diagonal indices")
		}
		if _, exists := state.Periods[key]; exists {
			panic("duplicate WPC recipe diagonal")
		}
		values := data[i*slots : (i+1)*slots]
		for j, value := range values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || value != values[j%period] {
				panic("WPC recipe values are nonfinite or not exactly periodic")
			}
		}
		state.Periods[key] = append([]float32(nil), values[:period]...)
		state.RecipeBytes += uint64(period * 4)
		state.MetadataBytes += 2 * 8 // normalized index and period length
	}
	state.FullBytes = uint64(len(indices) * (params.LevelQ + params.LevelP + 2) * 2 * slots * 8)
	return state
}

func (state *wpcOnlineRecipe) expand() lintrans.Diagonals[float64] {
	diagonals := make(lintrans.Diagonals[float64], len(state.Periods))
	for key, period := range state.Periods {
		values := make([]float64, state.Slots)
		for i := range values {
			values[i] = float64(period[i%len(period)])
		}
		diagonals[key] = values
	}
	return diagonals
}

//export GenerateWPCOnlineLinearTransform
func GenerateWPCOnlineLinearTransform(indicesC *C.int, indicesLen C.int, dataC *C.float, dataLen C.int, level C.int, ratio C.float, period C.int) C.int {
	indices := CArrayToSlice(indicesC, indicesLen, convertCIntToInt)
	data := CArrayToSlice(dataC, dataLen, func(value C.float) float32 { return float32(value) })
	params := lintrans.Parameters{
		DiagonalsIndexList: indices, LevelQ: int(level), LevelP: scheme.Params.MaxLevelP(),
		Scale:                     rlwe.NewScale(scheme.Params.Q()[int(level)]),
		LogDimensions:             ring.Dimensions{Rows: 0, Cols: scheme.Params.LogMaxSlots()},
		LogBabyStepGiantStepRatio: int(math.Log(float64(ratio))),
	}
	state := makeWPCOnlineRecipe(indices, data, int(period), params)
	id := AddLinearTransform(newLinearTransformationShell(params, 0))
	wpcOnlineMu.Lock()
	wpcOnlineRecipes[id] = state
	wpcOnlineMu.Unlock()
	return C.int(id)
}

//export EvaluateWPCOnlineLinearTransform
func EvaluateWPCOnlineLinearTransform(transformID, ciphertextID C.int) C.int {
	return C.int(evaluateWPCOnlineLinearTransform(int(transformID), int(ciphertextID)))
}

func evaluateWPCOnlineLinearTransform(transformID, ciphertextID int) int {
	wpcOnlineMu.Lock()
	state := wpcOnlineRecipes[int(transformID)]
	wpcOnlineMu.Unlock()
	if state == nil {
		panic("unknown WPC online recipe")
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	shell := RetrieveLinearTransform(int(transformID))
	// Restore empty polynomials even if Encode/evaluation panics. No full
	// transformation is retained between calls, and keys stay outside this path.
	wpcOnlineMu.Lock()
	wpcOnlineCurrentBytes += state.FullBytes
	wpcOnlineCurrentTransforms++
	if wpcOnlineCurrentBytes > wpcOnlinePeakBytes {
		wpcOnlinePeakBytes = wpcOnlineCurrentBytes
	}
	if wpcOnlineCurrentTransforms > wpcOnlinePeakTransforms {
		wpcOnlinePeakTransforms = wpcOnlineCurrentTransforms
	}
	wpcOnlineMu.Unlock()
	defer func() {
		for key := range shell.Vec {
			shell.Vec[key] = ringqp.Poly{}
		}
		wpcOnlineMu.Lock()
		wpcOnlineCurrentBytes -= state.FullBytes
		wpcOnlineCurrentTransforms--
		wpcOnlineMu.Unlock()
	}()
	started := time.Now()
	diagonals := state.expand()
	full := lintrans.NewTransformation(scheme.Params, state.Parameters)
	state.PrepareNS = uint64(time.Since(started).Nanoseconds())
	started = time.Now()
	if err := lintrans.Encode(scheme.Encoder, diagonals, full); err != nil {
		panic(err)
	}
	state.EncodeNS = uint64(time.Since(started).Nanoseconds())
	state.EncodeCalls++
	recordWPCBenchmarkEncode(2)
	for key, poly := range full.Vec {
		shell.Vec[key] = poly
	}
	started = time.Now()
	output := EvaluateLinearTransform(C.int(transformID), C.int(ciphertextID))
	state.EvaluateNS = uint64(time.Since(started).Nanoseconds())
	return int(output)
}

//export GetWPCOnlineLinearTransformStats
func GetWPCOnlineLinearTransformStats(transformID C.int) (*C.ulonglong, C.ulonglong) {
	wpcOnlineMu.Lock()
	state := wpcOnlineRecipes[int(transformID)]
	wpcOnlineMu.Unlock()
	if state == nil {
		return nil, 0
	}
	state.mu.Lock()
	values := []uint64{uint64(len(state.Periods)), state.FullBytes, state.RecipeBytes, state.MetadataBytes, state.PrepareNS, state.EncodeNS, state.EvaluateNS, state.EncodeCalls}
	state.mu.Unlock()
	result, length := SliceToCArray(values, convertUint64ToCULonglong)
	return result, C.ulonglong(length)
}

//export ResetWPCOnlineMaterializationPeak
func ResetWPCOnlineMaterializationPeak() {
	wpcOnlineMu.Lock()
	wpcOnlinePeakBytes = wpcOnlineCurrentBytes
	wpcOnlinePeakTransforms = wpcOnlineCurrentTransforms
	wpcOnlineMu.Unlock()
}

//export GetWPCOnlineGlobalStats
func GetWPCOnlineGlobalStats() (*C.ulonglong, C.ulonglong) {
	wpcOnlineMu.Lock()
	values := []uint64{uint64(len(wpcOnlineRecipes)), wpcOnlineCurrentBytes, wpcOnlinePeakBytes, wpcOnlineCurrentTransforms, wpcOnlinePeakTransforms}
	wpcOnlineMu.Unlock()
	result, length := SliceToCArray(values, convertUint64ToCULonglong)
	return result, C.ulonglong(length)
}

func deleteWPCOnlineRecipe(id int) {
	wpcOnlineMu.Lock()
	delete(wpcOnlineRecipes, id)
	wpcOnlineMu.Unlock()
}

func clearWPCOnlineRecipes() {
	wpcOnlineMu.Lock()
	wpcOnlineRecipes = make(map[int]*wpcOnlineRecipe)
	wpcOnlineCurrentBytes, wpcOnlinePeakBytes = 0, 0
	wpcOnlineCurrentTransforms, wpcOnlinePeakTransforms = 0, 0
	wpcOnlineMu.Unlock()
	ResetWPCBenchmarkEncodeCounters()
}
