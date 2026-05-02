package main

import (
	"C"

	"runtime"
	"runtime/debug"
	"time"

	"github.com/realqhc/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/realqhc/lattigo/v6/circuits/ckks/lintrans"
	"github.com/realqhc/lattigo/v6/circuits/ckks/polynomial"
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
}

//export DeleteScheme
func DeleteScheme() {
	scheme = Scheme{}

	DeleteRotationKeys()
	DeleteBootstrappers()
	DeleteMinimaxSignMap()

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
