package main

import (
	"testing"

	"github.com/realqhc/lattigo/v6/ring/ringqp"
	"github.com/stretchr/testify/require"
)

func TestWPCNativePayloadAccounting(t *testing.T) {
	params := wpcTestParameters(t)
	transform := wpcTestTransformation(params)
	n := uint64(params.N())
	q := n * uint64(params.MaxLevel()+1) * 8
	p := n * uint64(params.MaxLevelP()+1) * 8
	require.Equal(t, []uint64{1, 1, q, p, q + p}, linearTransformPayloadStats(transform))
	for key := range transform.Vec {
		transform.Vec[key] = ringqp.Poly{}
	}
	require.Equal(t, []uint64{1, 1, 0, 0, 0}, linearTransformPayloadStats(transform))
}
