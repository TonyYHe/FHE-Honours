package main

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/realqhc/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/realqhc/lattigo/v6/utils"
	"github.com/stretchr/testify/require"
)

func TestWPCPublicParameterManifest(t *testing.T) {
	previousScheme, previousBootstrappers := scheme, bootstrapperMap
	defer func() { scheme, bootstrapperMap = previousScheme, previousBootstrappers }()
	params := wpcTestParameters(t)
	scheme.Params = &params
	bootstrapperMap = make(map[int]*bootstrapping.Evaluator)
	manifest := wpcParameterManifest()
	require.False(t, manifest["security_assessed"].(bool))
	residual := manifest["residual"].(map[string]any)
	require.Equal(t, params.Q(), residual["q"])
	require.Equal(t, params.P(), residual["p"])
	require.Equal(t, params.LogQP(), residual["logqp"])
	require.Empty(t, manifest["bootstrappers"])
	btp, err := bootstrapping.NewParametersFromLiteral(params, bootstrapping.ParametersLiteral{
		LogN: utils.Pointy(params.LogN()), LogP: []int{61, 61}, Xs: params.Xs(),
		LogSlots: utils.Pointy(params.LogMaxSlots()),
	})
	require.NoError(t, err)
	// Metadata does not require generating keys or executing a bootstrap.
	bootstrapperMap[params.MaxSlots()] = &bootstrapping.Evaluator{Parameters: btp}
	manifest = wpcParameterManifest()
	rows := manifest["bootstrappers"].([]map[string]any)
	require.Len(t, rows, 1)
	boot := rows[0]["parameters"].(map[string]any)
	require.Equal(t, btp.BootstrappingParameters.Q(), boot["q"])
	require.Equal(t, btp.BootstrappingParameters.LogQP(), boot["logqp"])
	require.Greater(t, boot["logqp"].(float64), residual["logqp"].(float64))
	require.Equal(t, btp.EphemeralSecretWeight, rows[0]["ephemeral_secret_weight"])
	require.NotNil(t, rows[0]["encapsulation"])
	content, err := json.Marshal(manifest)
	require.NoError(t, err)
	require.False(t, strings.Contains(string(content), "SecretKey"))
	require.False(t, strings.Contains(string(content), "Coeffs"))
}
