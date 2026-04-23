from orion.backend.python.parameters import NewParameters


def test_empty_io_paths_remain_empty_strings() -> None:
    params = NewParameters(
        {
            "ckks_params": {
                "LogN": 12,
                "LogQ": [45, 30, 45],
                "LogP": [50],
                "LogScale": 30,
                "H": 64,
                "RingType": "Standard",
            },
            "orion": {
                "backend": "python",
                "io_mode": "save",
                "diags_path": "",
                "keys_path": "",
            },
        }
    )

    assert params.get_diags_path() == ""
    assert params.get_keys_path() == ""
    assert params.io_paths_exist() is False
