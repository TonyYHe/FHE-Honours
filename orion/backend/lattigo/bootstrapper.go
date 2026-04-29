package main

import (
	"C"
)
import (
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"

	"github.com/realqhc/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/realqhc/lattigo/v6/core/rlwe"
	"github.com/realqhc/lattigo/v6/utils"
)

// Map to store bootstrapping.Evaluators by their slot count
// Initialize the map at package level
var bootstrapperMap = make(map[int]*bootstrapping.Evaluator)

//export NewBootstrapper
func NewBootstrapper(
	LogPs *C.int,
	lenLogPs C.int,
	numSlots C.int,
) {
	slots := int(numSlots)

	if _, exists := bootstrapperMap[slots]; exists {
		return
	}

	// If not initialized for this slot count, create a new one
	logP := CArrayToSlice(LogPs, lenLogPs, convertCIntToInt)

	btpParametersLit := bootstrapping.ParametersLiteral{
		LogN:     utils.Pointy(scheme.Params.LogN()),
		LogP:     logP,
		Xs:       scheme.Params.Xs(),
		LogSlots: utils.Pointy(int(math.Log2(float64(slots)))),
	}

	btpParams, err := bootstrapping.NewParametersFromLiteral(
		*scheme.Params, btpParametersLit)
	if err != nil {
		panic(err)
	}

	btpKeys, _, err := btpParams.GenEvaluationKeys(scheme.SecretKey)
	if err != nil {
		panic(err)
	}

	var btpEval *bootstrapping.Evaluator
	if btpEval, err = bootstrapping.NewEvaluator(btpParams, btpKeys); err != nil {
		panic(err)
	}

	// Store the new evaluator in the map
	bootstrapperMap[slots] = btpEval
}

//export Bootstrap
func Bootstrap(ciphertextID, numSlots C.int) C.int {
	ctIn := RetrieveCiphertext(int(ciphertextID))
	bootstrapper := GetBootstrapper(int(numSlots))

	ctOut, err := bootstrapOne(bootstrapper, ctIn)
	if err != nil {
		panic(err)
	}

	idx := PushCiphertext(ctOut)
	return C.int(idx)
}

//export BootstrapMany
func BootstrapMany(ciphertextIDs *C.int, lenCiphertextIDs C.int, numSlots C.int) (*C.int, C.ulong) {
	ids := CArrayToSlice(ciphertextIDs, lenCiphertextIDs, convertCIntToInt)
	if len(ids) == 0 {
		return nil, 0
	}

	bootstrapper := GetBootstrapper(int(numSlots))
	inputs := make([]*rlwe.Ciphertext, len(ids))
	for i, id := range ids {
		inputs[i] = RetrieveCiphertext(id)
	}

	ctsOut, err := bootstrapMany(bootstrapper, inputs)
	if err != nil {
		panic(err)
	}
	if len(ctsOut) != len(ids) {
		panic(fmt.Errorf("BootstrapMany returned %d ciphertexts for %d inputs", len(ctsOut), len(ids)))
	}

	outIDs := make([]int, len(ctsOut))
	for i := range ctsOut {
		outIDs[i] = PushCiphertext(ctsOut[i])
	}

	return SliceToCArray(outIDs, convertIntToCInt)
}

func bootstrapOne(bootstrapper *bootstrapping.Evaluator, ctIn *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	ctBtp := ctIn.CopyNew()
	ctBtp.LogDimensions.Cols = bootstrapper.LogMaxSlots()

	ctOut, err := bootstrapper.Bootstrap(ctBtp)
	if err != nil {
		return nil, err
	}

	postscale := int(1 << (scheme.Params.LogMaxSlots() - bootstrapper.LogMaxSlots()))
	if err := bootstrapper.Evaluator.Mul(ctOut, postscale, ctOut); err != nil {
		return nil, err
	}

	ctOut.LogDimensions.Cols = scheme.Params.LogMaxSlots()
	return ctOut, nil
}

func bootstrapMany(bootstrapper *bootstrapping.Evaluator, inputs []*rlwe.Ciphertext) ([]*rlwe.Ciphertext, error) {
	workers := bootstrapWorkerCount(len(inputs))
	if workers <= 1 || len(inputs) <= 1 {
		outputs := make([]*rlwe.Ciphertext, len(inputs))
		for i, ctIn := range inputs {
			ctOut, err := bootstrapOne(bootstrapper, ctIn)
			if err != nil {
				return nil, err
			}
			outputs[i] = ctOut
		}
		return outputs, nil
	}

	outputs := make([]*rlwe.Ciphertext, len(inputs))
	jobs := make(chan int)
	var wg sync.WaitGroup
	var errMu sync.Mutex
	var firstErr error

	setErr := func(err error) {
		if err == nil {
			return
		}
		errMu.Lock()
		defer errMu.Unlock()
		if firstErr == nil {
			firstErr = err
		}
	}

	for workerID := 0; workerID < workers; workerID++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for index := range jobs {
				ctOut, err := bootstrapOne(bootstrapper, inputs[index])
				if err != nil {
					setErr(err)
					continue
				}
				outputs[index] = ctOut
			}
		}()
	}

	for index := range inputs {
		jobs <- index
	}
	close(jobs)
	wg.Wait()

	if firstErr != nil {
		return nil, firstErr
	}
	for i, ctOut := range outputs {
		if ctOut == nil {
			return nil, fmt.Errorf("BootstrapMany produced nil ciphertext at index %d", i)
		}
	}
	return outputs, nil
}

func bootstrapWorkerCount(numInputs int) int {
	if numInputs <= 1 {
		return 1
	}

	raw := strings.TrimSpace(os.Getenv("ORION_LATTIGO_BOOTSTRAP_WORKERS"))
	if raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil {
			panic(fmt.Errorf("invalid ORION_LATTIGO_BOOTSTRAP_WORKERS=%q", raw))
		}
		if parsed <= 1 {
			return 1
		}
		return min(parsed, numInputs, runtime.GOMAXPROCS(0))
	}

	limit := min(numInputs, runtime.GOMAXPROCS(0), 4)
	if limit < 1 {
		return 1
	}
	return limit
}

func GetBootstrapper(numSlots int) *bootstrapping.Evaluator {
	bootstrapper, exists := bootstrapperMap[numSlots]
	if !exists {
		panic(fmt.Errorf("no bootstrapper found for slot count: %d", numSlots))
	}
	return bootstrapper
}

//export DeleteBootstrappers
func DeleteBootstrappers() {
	bootstrapperMap = make(map[int]*bootstrapping.Evaluator)
}
