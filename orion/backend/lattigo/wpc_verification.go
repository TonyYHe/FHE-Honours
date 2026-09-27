package main

/*
#include <stdint.h>
*/
import "C"

import (
	"slices"
	"unsafe"

	lintrans "github.com/realqhc/lattigo/v6/circuits/ckks/lintrans"
	"github.com/realqhc/lattigo/v6/core/rlwe"
	"github.com/realqhc/lattigo/v6/ring"
	"github.com/realqhc/lattigo/v6/ring/ringqp"
	"github.com/realqhc/lattigo/v6/schemes/ckks"
)

const (
	wpcVerificationPassed            = 1
	wpcVerificationEncodedMismatch   = 0
	wpcVerificationInvalidInput      = -1
	wpcVerificationSourceNotPeriodic = -2
	wpcVerificationEncodeFailure     = -3
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

func exactEvaluationRoundTrip(poly ring.Poly, evaluationPeriod int) bool {
	for _, coeffs := range poly.Coeffs {
		representatives, periodic := evaluationBlockRepresentatives(coeffs, evaluationPeriod)
		if !periodic {
			return false
		}
		reconstructed, ok := reconstructEvaluationBlocks(representatives, len(coeffs))
		if !ok || !slices.Equal(coeffs, reconstructed) {
			return false
		}
	}
	return true
}

func exactQPEvaluationRoundTrip(poly ringqp.Poly, evaluationPeriod int) bool {
	return exactEvaluationRoundTrip(poly.Q, evaluationPeriod) &&
		exactEvaluationRoundTrip(poly.P, evaluationPeriod)
}

func exactRealSlotPeriod(values []float64, period int) bool {
	if period <= 0 || period >= len(values) || len(values)%period != 0 {
		return false
	}
	for i := period; i < len(values); i++ {
		if values[i] != values[i%period] {
			return false
		}
	}
	return true
}

func exactComplexSlotPeriod(values []complex128, period int) bool {
	if period <= 0 || period >= len(values) || len(values)%period != 0 {
		return false
	}
	for i := period; i < len(values); i++ {
		if values[i] != values[i%period] {
			return false
		}
	}
	return true
}

func wpcVerificationTransformation(params ckks.Parameters, levelQ int) lintrans.LinearTransformation {
	return lintrans.NewTransformation(params, lintrans.Parameters{
		DiagonalsIndexList:        []int{0},
		LevelQ:                    levelQ,
		LevelP:                    params.MaxLevelP(),
		Scale:                     rlwe.NewScale(params.Q()[levelQ]),
		LogDimensions:             ring.Dimensions{Rows: 0, Cols: params.LogMaxSlots()},
		LogBabyStepGiantStepRatio: -1,
	})
}

func verifyWPCEncodedReal(params ckks.Parameters, values []float64, levelQ, slotPeriod int) int {
	if len(values) != params.MaxSlots() || levelQ < 0 || levelQ > params.MaxLevel() {
		return wpcVerificationInvalidInput
	}
	if !exactRealSlotPeriod(values, slotPeriod) {
		return wpcVerificationSourceNotPeriodic
	}
	transform := wpcVerificationTransformation(params, levelQ)
	if err := lintrans.Encode(
		ckks.NewEncoder(params),
		lintrans.Diagonals[float64]{0: values},
		transform,
	); err != nil {
		return wpcVerificationEncodeFailure
	}
	if !exactQPEvaluationRoundTrip(transform.Vec[0], 2*slotPeriod) {
		return wpcVerificationEncodedMismatch
	}
	return wpcVerificationPassed
}

func verifyWPCEncodedComplex(params ckks.Parameters, values []complex128, levelQ, slotPeriod int) int {
	if len(values) != params.MaxSlots() || levelQ < 0 || levelQ > params.MaxLevel() {
		return wpcVerificationInvalidInput
	}
	if !exactComplexSlotPeriod(values, slotPeriod) {
		return wpcVerificationSourceNotPeriodic
	}
	transform := wpcVerificationTransformation(params, levelQ)
	if err := lintrans.Encode(
		ckks.NewEncoder(params),
		lintrans.Diagonals[complex128]{0: values},
		transform,
	); err != nil {
		return wpcVerificationEncodeFailure
	}
	if !exactQPEvaluationRoundTrip(transform.Vec[0], 2*slotPeriod) {
		return wpcVerificationEncodedMismatch
	}
	return wpcVerificationPassed
}

// VerifyWPCEncodedDiagonal independently encodes one slot-periodic candidate
// with the active CKKS parameters, verifies the expected 2T evaluation-form
// copy map in every Q/P limb, and reconstructs every coefficient exactly.
//
//export VerifyWPCEncodedDiagonal
func VerifyWPCEncodedDiagonal(
	valuesC *C.double,
	valuesLen C.int,
	isComplex C.int,
	levelQ C.int,
	slotPeriod C.int,
) C.int {
	if scheme.Params == nil || valuesC == nil || int(valuesLen) <= 0 {
		return C.int(wpcVerificationInvalidInput)
	}
	raw := unsafe.Slice(valuesC, int(valuesLen))
	if int(isComplex) == 0 {
		values := make([]float64, len(raw))
		for i := range raw {
			values[i] = float64(raw[i])
		}
		return C.int(verifyWPCEncodedReal(
			*scheme.Params,
			values,
			int(levelQ),
			int(slotPeriod),
		))
	}
	if len(raw)%2 != 0 {
		return C.int(wpcVerificationInvalidInput)
	}
	values := make([]complex128, len(raw)/2)
	for i := range values {
		values[i] = complex(float64(raw[2*i]), float64(raw[2*i+1]))
	}
	return C.int(verifyWPCEncodedComplex(
		*scheme.Params,
		values,
		int(levelQ),
		int(slotPeriod),
	))
}
