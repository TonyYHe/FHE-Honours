from __future__ import annotations

from types import SimpleNamespace

from orion.experimental.cir.runtime_group import RegionFirstRuntimeGroup


def test_region_runtime_group_executes_once_per_source_ciphertext() -> None:
    calls = []

    def executor(source):
        calls.append(tuple(source.ids))
        return {"a": f"a:{source.ids[0]}", "b": f"b:{source.ids[0]}"}

    group = RegionFirstRuntimeGroup(
        region_id="r",
        network="R18",
        stage="stage1",
        module_prefix="layers.0",
        conv_nodes=("a", "b"),
        strategy="test",
        materializer="test",
        depth=2,
        boundary_actions=("insert_extract_before_relu_or_add",),
        expected_stats={},
        executable=True,
        fallback_reason="",
        executor=executor,
    )

    source = SimpleNamespace(ids=[11])

    assert group.output("a", source) == "a:11"
    assert group.output("b", source) == "b:11"
    assert calls == [(11,)]
    assert group.execute_count == 1

    assert group.output("a", SimpleNamespace(ids=[12])) == "a:12"
    assert calls == [(11,), (12,)]
    assert group.execute_count == 2


def test_region_runtime_group_rejects_non_executable_group() -> None:
    group = RegionFirstRuntimeGroup(
        region_id="r",
        network="R18",
        stage="stage1",
        module_prefix="layers.0",
        conv_nodes=("a",),
        strategy="test",
        materializer="test",
        depth=2,
        boundary_actions=(),
        expected_stats={},
        executable=False,
        fallback_reason="missing_fused_weight_materializer",
    )

    try:
        group.output("a", SimpleNamespace(ids=[1]))
    except RuntimeError as exc:
        assert "missing_fused_weight_materializer" in str(exc)
    else:
        raise AssertionError("expected non-executable runtime group to raise")


def test_executable_region_runtime_group_requires_executor() -> None:
    group = RegionFirstRuntimeGroup(
        region_id="r",
        network="R18",
        stage="stage1",
        module_prefix="layers.0",
        conv_nodes=("a",),
        strategy="test",
        materializer="test",
        depth=2,
        boundary_actions=(),
        expected_stats={},
        executable=True,
        fallback_reason="",
    )

    try:
        group.output("a", SimpleNamespace(ids=[1]))
    except RuntimeError as exc:
        assert "has no executor" in str(exc)
    else:
        raise AssertionError("expected executable group without executor to raise")
