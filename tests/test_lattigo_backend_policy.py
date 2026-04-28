from orion.backend.lattigo.bindings import _read_env_bool, _read_env_int


def test_lattigo_policy_env_bool_uses_aliases(monkeypatch):
    monkeypatch.delenv("ORION_PRIMARY_FLAG", raising=False)
    monkeypatch.setenv("ORION_ALIAS_FLAG", "0")

    assert not _read_env_bool(("ORION_PRIMARY_FLAG", "ORION_ALIAS_FLAG"), True)


def test_lattigo_policy_env_bool_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("ORION_UNSET_FLAG", raising=False)

    assert _read_env_bool(("ORION_UNSET_FLAG",), True)
    assert not _read_env_bool(("ORION_UNSET_FLAG",), False)


def test_lattigo_policy_env_int_ignores_invalid_values(monkeypatch):
    monkeypatch.setenv("ORION_INT_FLAG", "not-an-int")

    assert _read_env_int("ORION_INT_FLAG") is None


def test_lattigo_policy_env_int_reads_integer(monkeypatch):
    monkeypatch.setenv("ORION_INT_FLAG", "4096")

    assert _read_env_int("ORION_INT_FLAG") == 4096
