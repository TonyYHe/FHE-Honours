from __future__ import annotations

from types import SimpleNamespace

from orion.backend.python import compile_cache


class _FakeExecutor:
    def __init__(self) -> None:
        self.loaded_metadata = None

    def load_compile_cache_metadata(self, metadata):
        self.loaded_metadata = dict(metadata)


def _node(executor: _FakeExecutor):
    return {"module": SimpleNamespace(region_runtime=SimpleNamespace(executor=executor))}


def test_apply_provider_metadata_infers_legacy_tconv_storage_keys() -> None:
    up_executor = _FakeExecutor()
    dag = SimpleNamespace(
        nodes={
            "before": _node(_FakeExecutor()),
            "up3": _node(up_executor),
            "after": _node(_FakeExecutor()),
        }
    )
    provider_metadata = {
        "rows": [
            {
                "node": "before",
                "executor_type": "orion.experimental.cir.r34_orion_same_shape.R34InterGroupHybridSameShapeRuntimeExecutor",
                "executor_metadata": {"groups_by_input_block": [{"storage_key": "group_150"}]},
            },
            {
                "node": "up3",
                "executor_type": "orion.experimental.u22_phase1.TconvK2S2PythonRuntimeExecutor",
                "executor_metadata": {},
            },
            {
                "node": "after",
                "executor_type": "orion.experimental.cir.r34_orion_same_shape.R34InterGroupHybridSameShapeRuntimeExecutor",
                "executor_metadata": {"groups_by_input_block": [{"storage_key": "group_155"}]},
            },
        ]
    }

    compile_cache.apply_provider_metadata(dag, provider_metadata)

    assert up_executor.loaded_metadata == {
        "kind": "TconvK2S2PythonRuntimeExecutor",
        "inferred_storage_keys": ["group_151", "group_152", "group_153", "group_154"],
        "inferred_from_neighbor_group_gap": True,
    }
