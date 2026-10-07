package main

import "C"

import (
	"encoding/json"
	"sort"

	"github.com/realqhc/lattigo/v6/ring"
	"github.com/realqhc/lattigo/v6/schemes/ckks"
)

// Public parameter metadata only: no key values or secret coefficients.
func wpcDistributionManifest(distribution ring.DistributionParameters) map[string]any {
	// Avoid distribution.MarshalJSON's omitted zero fields and embedded Type.
	var parameters any
	switch value := distribution.(type) {
	case ring.Ternary:
		parameters = map[string]any{"P": value.P, "H": value.H}
	case ring.DiscreteGaussian:
		parameters = map[string]any{"Sigma": value.Sigma, "Bound": value.Bound}
	default:
		parameters = distribution
	}
	return map[string]any{"type": distribution.Type(), "parameters": parameters}
}

func wpcCKKSParameterManifest(params ckks.Parameters) map[string]any {
	return map[string]any{
		"logn": params.LogN(), "ring_type": params.RingType().String(),
		"q": params.Q(), "p": params.P(), "logqp": params.LogQP(),
		"log_default_scale":   params.LogDefaultScale(),
		"secret_distribution": wpcDistributionManifest(params.Xs()),
		"error_distribution":  wpcDistributionManifest(params.Xe()),
	}
}

func wpcParameterManifest() map[string]any {
	slots := make([]int, 0, len(bootstrapperMap))
	for slotCount := range bootstrapperMap {
		slots = append(slots, slotCount)
	}
	sort.Ints(slots)
	bootstrappers := make([]map[string]any, 0, len(slots))
	for _, slotCount := range slots {
		params := bootstrapperMap[slotCount].Parameters
		var encapsulation any
		if params.EphemeralSecretWeight != 0 {
			// Matches genEncapsulationEvaluationKeysNew in the bundled fork.
			encapsulation = map[string]any{
				"q":                     params.BootstrappingParameters.Q()[:1],
				"p":                     params.BootstrappingParameters.P()[:1],
				"secret_hamming_weight": params.EphemeralSecretWeight,
			}
		}
		bootstrappers = append(bootstrappers, map[string]any{
			"slots":                   slotCount,
			"parameters":              wpcCKKSParameterManifest(params.BootstrappingParameters),
			"ephemeral_secret_weight": params.EphemeralSecretWeight,
			"encapsulation":           encapsulation,
		})
	}
	return map[string]any{
		"schema_version": 1, "security_assessed": false,
		"residual": wpcCKKSParameterManifest(*scheme.Params), "bootstrappers": bootstrappers,
	}
}

// GetWPCParameterManifest returns UTF-8 JSON octets as an owned uint64 array.
// This uses the existing auto-freed ctypes array ABI (no C string lifetime).
//
//export GetWPCParameterManifest
func GetWPCParameterManifest() (*C.ulonglong, C.ulong) {
	content, err := json.Marshal(wpcParameterManifest())
	if err != nil {
		panic(err)
	}
	octets := make([]uint64, len(content))
	for index, value := range content {
		octets[index] = uint64(value)
	}
	return SliceToCArray(octets, convertUint64ToCULonglong)
}
