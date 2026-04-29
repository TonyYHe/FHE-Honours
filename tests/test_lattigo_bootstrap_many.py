from __future__ import annotations

from types import SimpleNamespace

import torch

from orion.backend.python.tensors import CipherTensor


class _FakeBackend:
    def DeleteCiphertext(self, _idx):
        return None

    def GetCiphertextSlots(self, _idx):
        return 16


class _FakeBootstrapper:
    def __init__(self):
        self.single_calls = []
        self.many_calls = []

    def bootstrap(self, ctxt, slots):
        self.single_calls.append((int(ctxt), int(slots)))
        return int(ctxt) + 1000

    def bootstrap_many(self, ctxts, slots):
        self.many_calls.append(([int(ctxt) for ctxt in ctxts], int(slots)))
        return [int(ctxt) + 100 for ctxt in ctxts]


def _cipher_tensor(bootstrapper: _FakeBootstrapper) -> CipherTensor:
    scheme = SimpleNamespace(
        backend=_FakeBackend(),
        encryptor=SimpleNamespace(),
        evaluator=SimpleNamespace(),
        bootstrapper=bootstrapper,
    )
    return CipherTensor(
        scheme,
        [1, 2, 3],
        torch.Size([3]),
        torch.Size([3]),
    )


def test_cipher_tensor_bootstrap_uses_single_path_by_default(monkeypatch) -> None:
    monkeypatch.delenv("ORION_LATTIGO_BOOTSTRAP_MANY", raising=False)
    bootstrapper = _FakeBootstrapper()

    out = _cipher_tensor(bootstrapper).bootstrap(slots=8)

    assert out.ids == [1001, 1002, 1003]
    assert bootstrapper.many_calls == []
    assert bootstrapper.single_calls == [(1, 8), (2, 8), (3, 8)]


def test_cipher_tensor_bootstrap_can_enable_many_path(monkeypatch) -> None:
    monkeypatch.setenv("ORION_LATTIGO_BOOTSTRAP_MANY", "1")
    bootstrapper = _FakeBootstrapper()

    out = _cipher_tensor(bootstrapper).bootstrap(slots=8)

    assert out.ids == [101, 102, 103]
    assert bootstrapper.many_calls == [([1, 2, 3], 8)]
    assert bootstrapper.single_calls == []


def test_cipher_tensor_bootstrap_can_disable_many_path(monkeypatch) -> None:
    monkeypatch.setenv("ORION_LATTIGO_BOOTSTRAP_MANY", "0")
    bootstrapper = _FakeBootstrapper()

    out = _cipher_tensor(bootstrapper).bootstrap(slots=8)

    assert out.ids == [1001, 1002, 1003]
    assert bootstrapper.many_calls == []
    assert bootstrapper.single_calls == [(1, 8), (2, 8), (3, 8)]
