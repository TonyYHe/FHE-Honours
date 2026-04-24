import ctypes
import os
import platform

from orion.backend.lattigo.bindings import LattigoLibrary


class CheddarLibrary(LattigoLibrary):
    """ctypes loader for the Orion-side Cheddar compatibility wrapper."""

    def __init__(self):
        super().__init__()
        self.lt_outputs_are_rescaled = True

    def _load_library(self):
        try:
            if platform.system() == "Linux":
                lib_name = "cheddar-linux.so"
            elif platform.system() == "Darwin":
                if platform.machine().lower() in ("arm64", "aarch64"):
                    lib_name = "cheddar-mac-arm64.dylib"
                else:
                    lib_name = "cheddar-mac.dylib"
            elif platform.system() == "Windows":
                lib_name = "cheddar-windows.dll"
            else:
                raise RuntimeError("Unsupported platform")

            current_dir = os.path.dirname(os.path.abspath(__file__))
            lib_path = os.path.join(current_dir, lib_name)
            return ctypes.CDLL(lib_path)
        except OSError as e:
            raise RuntimeError(f"Failed to load Cheddar library: {e}")
