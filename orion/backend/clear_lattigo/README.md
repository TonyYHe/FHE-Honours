# Clear Lattigo Backend

This directory contains an opt-in non-CKKS backend that exports the same C ABI
as `orion.backend.lattigo`. It is intended for fast correctness and path-smoke
tests of the Lattigo/provider/native-halo lowering, not for CKKS timing,
security, or noise evidence.

Build locally:

```bash
c++ -std=c++17 -O3 -shared -fPIC \
  orion/backend/clear_lattigo/clear_lattigo.cpp \
  -o orion/backend/clear_lattigo/clear_lattigo-linux.so
```

Use either:

- `ORION_LATTIGO_CLEAR_BACKEND=1` with normal `backend: lattigo`, which keeps
  provider allowlists on the canonical Lattigo path.
- `backend: clear_lattigo` for explicit tests.
