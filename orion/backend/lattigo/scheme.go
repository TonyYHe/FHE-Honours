package main

import (
	"C"

	"runtime"
	"runtime/debug"
	"sync/atomic"
	"time"

	"github.com/realqhc/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/realqhc/lattigo/v6/circuits/ckks/lintrans"
	"github.com/realqhc/lattigo/v6/circuits/ckks/polynomial"
	commonlintrans "github.com/realqhc/lattigo/v6/circuits/common/lintrans"
	"github.com/realqhc/lattigo/v6/core/rlwe"
	"github.com/realqhc/lattigo/v6/ring"
	"github.com/realqhc/lattigo/v6/schemes/ckks"
)

type Scheme struct {
	Params        *ckks.Parameters
	KeyGen        *rlwe.KeyGenerator
	SecretKey     *rlwe.SecretKey
	PublicKey     *rlwe.PublicKey
	RelinKey      *rlwe.RelinearizationKey
	EvalKeys      *rlwe.MemEvaluationKeySet
	Encoder       *ckks.Encoder
	Encryptor     *rlwe.Encryptor
	Decryptor     *rlwe.Decryptor
	Evaluator     *ckks.Evaluator
	PolyEvaluator *polynomial.Evaluator
	LinEvaluator  *lintrans.Evaluator
	Bootstrapper  *bootstrapping.Evaluator
}

var scheme Scheme
var runtimeMemoryTrimSeconds float64
var opRotationCount uint64
var opLintransRotationCount uint64
var opDirectRotationCount uint64
var opConjugationCount uint64

func installOperationCallbacks() {
	commonlintrans.SetOperationCallbacks(&commonlintrans.OperationCallbacks{
		OnRotation: func(_ int) {
			atomic.AddUint64(&opRotationCount, 1)
			atomic.AddUint64(&opLintransRotationCount, 1)
		},
	})
	ckks.SetOperationCallbacks(&ckks.OperationCallbacks{
		OnRotation: func(_ int) {
			atomic.AddUint64(&opRotationCount, 1)
			atomic.AddUint64(&opDirectRotationCount, 1)
		},
		OnConjugation: func() {
			atomic.AddUint64(&opConjugationCount, 1)
		},
	})
}

//export ResetOperationCounters
func ResetOperationCounters() {
	atomic.StoreUint64(&opRotationCount, 0)
	atomic.StoreUint64(&opLintransRotationCount, 0)
	atomic.StoreUint64(&opDirectRotationCount, 0)
	atomic.StoreUint64(&opConjugationCount, 0)
	installOperationCallbacks()
}

//export GetOperationCounters
func GetOperationCounters() (*C.ulonglong, C.ulonglong) {
	values := []uint64{
		atomic.LoadUint64(&opRotationCount),
		atomic.LoadUint64(&opLintransRotationCount),
		atomic.LoadUint64(&opDirectRotationCount),
		atomic.LoadUint64(&opConjugationCount),
	}
	arrPtr, length := SliceToCArray(values, convertUint64ToCULonglong)
	return arrPtr, C.ulonglong(length)
}

func trimRuntimeMemory() float64 {
	started := time.Now()
	runtime.GC()
	debug.FreeOSMemory()
	elapsed := time.Since(started).Seconds()
	runtimeMemoryTrimSeconds += elapsed
	return elapsed
}

//export NewScheme
func NewScheme(
	logN C.int,
	logQPtr *C.int, lenQ C.int,
	logPPtr *C.int, lenP C.int,
	logScale C.int,
	h C.int,
	ringType *C.char,
	keysPath *C.char,
	ioMode *C.char,
) {
	// Convert LogQ and LogP to Go slices
	logQ := CArrayToSlice(logQPtr, lenQ, convertCIntToInt)
	logP := CArrayToSlice(logPPtr, lenP, convertCIntToInt)

	ringT := ring.Standard
	if C.GoString(ringType) != "standard" {
		ringT = ring.ConjugateInvariant
	}

	var err error
	var params ckks.Parameters

	if params, err = ckks.NewParametersFromLiteral(
		ckks.ParametersLiteral{
			LogN:            int(logN),
			LogQ:            logQ,
			LogP:            logP,
			LogDefaultScale: int(logScale),
			Xs:              ring.Ternary{H: int(h)},
			RingType:        ringT,
		}); err != nil {
		panic(err)
	}

	keyGen := rlwe.NewKeyGenerator(params)

	scheme = Scheme{
		Params:        &params,
		KeyGen:        keyGen,
		SecretKey:     nil,
		PublicKey:     nil,
		RelinKey:      nil,
		EvalKeys:      nil,
		Encoder:       nil,
		Encryptor:     nil,
		Decryptor:     nil,
		Evaluator:     nil,
		PolyEvaluator: nil,
		LinEvaluator:  nil,
		Bootstrapper:  nil,
	}
	ResetOperationCounters()
	resetSharedCacheEvalProfile()
}

//export DeleteScheme
func DeleteScheme() {
	scheme = Scheme{}
	commonlintrans.ClearOperationCallbacks()
	ckks.ClearOperationCallbacks()
	resetSharedCacheEvalProfile()

	DeleteRotationKeys()
	DeleteBootstrappers()
	DeleteMinimaxSignMap()
	clearPredecodedLinearTransformArtifacts()

	ltHeap.Reset()
	polyHeap.Reset()
	ptHeap.Reset()
	ctHeap.Reset()
	trimRuntimeMemory()
}

//export TrimRuntimeMemory
func TrimRuntimeMemory() C.double {
	return C.double(trimRuntimeMemory())
}

//export ConsumeRuntimeMemoryTrimSeconds
func ConsumeRuntimeMemoryTrimSeconds() C.double {
	elapsed := runtimeMemoryTrimSeconds
	runtimeMemoryTrimSeconds = 0
	return C.double(elapsed)
}

//export GetRuntimeMemoryStats
func GetRuntimeMemoryStats() (*C.ulonglong, C.ulonglong) {
	var stats runtime.MemStats
	runtime.ReadMemStats(&stats)
	values := []uint64{
		stats.Alloc,
		stats.TotalAlloc,
		stats.Sys,
		stats.HeapAlloc,
		stats.HeapSys,
		stats.HeapIdle,
		stats.HeapReleased,
		stats.HeapInuse,
		stats.StackInuse,
		stats.MSpanInuse,
		stats.MCacheInuse,
		uint64(stats.NumGC),
	}
	arrPtr, length := SliceToCArray(values, convertUint64ToCULonglong)
	return arrPtr, C.ulonglong(length)
}
