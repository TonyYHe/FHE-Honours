#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

extern "C" {

struct ArrayResultInt {
  int *Data;
  unsigned long Length;
};

struct ArrayResultFloat {
  float *Data;
  unsigned long Length;
};

struct ArrayResultDouble {
  double *Data;
  unsigned long Length;
};

struct ArrayResultUInt64 {
  unsigned long long *Data;
  unsigned long long Length;
};

struct ArrayResultByte {
  unsigned char *Data;
  unsigned long Length;
};

}

namespace {

using Complex = std::complex<double>;

struct TensorState {
  std::vector<Complex> values;
  int level = 0;
  uint64_t scale = 1;
  int degree = 0;
};

struct LinearTransformState {
  std::vector<int> diag_indices;
  std::vector<std::vector<Complex>> diagonals;
  int level = 0;
  int slots = 0;
};

struct PolynomialState {
  enum class Kind { Monomial, Chebyshev };
  Kind kind = Kind::Monomial;
  std::vector<double> coeffs;
};

struct ClearScheme {
  int logn = 0;
  std::vector<int> logq;
  std::vector<int> logp;
  int logscale = 0;
  std::string ring_type = "standard";
  int slots = 0;
  int next_id = 1;
  std::map<int, TensorState> plaintexts;
  std::map<int, TensorState> ciphertexts;
  std::map<int, LinearTransformState> linear_transforms;
  std::map<int, PolynomialState> polynomials;
  std::set<int> rotation_keys;
  uint64_t rotations_total = 0;
  uint64_t rotations_lt = 0;
  uint64_t rotations_direct = 0;
  uint64_t conjugations = 0;
};

std::mutex g_mu;
ClearScheme g_scheme;

[[noreturn]] void AbortWithMessage(const char *message) {
  std::fprintf(stderr, "%s\n", message);
  std::abort();
}

int AllocateId() { return g_scheme.next_id++; }

uint64_t ModulusProxyFromLog(int log_value) {
  if (log_value <= 0) {
    return 1;
  }
  if (log_value >= 63) {
    return uint64_t{1} << 62;
  }
  return uint64_t{1} << static_cast<unsigned>(log_value);
}

int MaxLevel() { return std::max(0, static_cast<int>(g_scheme.logq.size()) - 1); }

TensorState &Plaintext(int id) {
  auto it = g_scheme.plaintexts.find(id);
  if (it == g_scheme.plaintexts.end()) {
    AbortWithMessage("clear_lattigo plaintext id not found");
  }
  return it->second;
}

TensorState &Ciphertext(int id) {
  auto it = g_scheme.ciphertexts.find(id);
  if (it == g_scheme.ciphertexts.end()) {
    AbortWithMessage("clear_lattigo ciphertext id not found");
  }
  return it->second;
}

LinearTransformState &LinearTransform(int id) {
  auto it = g_scheme.linear_transforms.find(id);
  if (it == g_scheme.linear_transforms.end()) {
    AbortWithMessage("clear_lattigo linear transform id not found");
  }
  return it->second;
}

PolynomialState &Polynomial(int id) {
  auto it = g_scheme.polynomials.find(id);
  if (it == g_scheme.polynomials.end()) {
    AbortWithMessage("clear_lattigo polynomial id not found");
  }
  return it->second;
}

template <typename T>
T *AllocArray(std::size_t count) {
  if (count == 0) {
    return nullptr;
  }
  void *ptr = std::malloc(sizeof(T) * count);
  if (ptr == nullptr) {
    AbortWithMessage("clear_lattigo malloc failed");
  }
  return static_cast<T *>(ptr);
}

ArrayResultInt MakeIntArrayResult(const std::vector<int> &values) {
  ArrayResultInt result{nullptr, 0};
  if (values.empty()) {
    return result;
  }
  result.Data = AllocArray<int>(values.size());
  result.Length = static_cast<unsigned long>(values.size());
  std::memcpy(result.Data, values.data(), sizeof(int) * values.size());
  return result;
}

ArrayResultFloat MakeFloatArrayResult(const std::vector<float> &values) {
  ArrayResultFloat result{nullptr, 0};
  if (values.empty()) {
    return result;
  }
  result.Data = AllocArray<float>(values.size());
  result.Length = static_cast<unsigned long>(values.size());
  std::memcpy(result.Data, values.data(), sizeof(float) * values.size());
  return result;
}

ArrayResultDouble MakeDoubleArrayResult(const std::vector<double> &values) {
  ArrayResultDouble result{nullptr, 0};
  if (values.empty()) {
    return result;
  }
  result.Data = AllocArray<double>(values.size());
  result.Length = static_cast<unsigned long>(values.size());
  std::memcpy(result.Data, values.data(), sizeof(double) * values.size());
  return result;
}

ArrayResultUInt64 MakeUInt64ArrayResult(const std::vector<unsigned long long> &values) {
  ArrayResultUInt64 result{nullptr, 0};
  if (values.empty()) {
    return result;
  }
  result.Data = AllocArray<unsigned long long>(values.size());
  result.Length = static_cast<unsigned long long>(values.size());
  std::memcpy(result.Data, values.data(), sizeof(unsigned long long) * values.size());
  return result;
}

ArrayResultByte MakeByteArrayResult(const std::vector<unsigned char> &values) {
  ArrayResultByte result{nullptr, 0};
  if (values.empty()) {
    return result;
  }
  result.Data = AllocArray<unsigned char>(values.size());
  result.Length = static_cast<unsigned long>(values.size());
  std::memcpy(result.Data, values.data(), values.size());
  return result;
}

std::vector<int> ReadIntArray(const int *data, int len) {
  std::vector<int> out;
  if (data == nullptr || len <= 0) {
    return out;
  }
  out.assign(data, data + len);
  return out;
}

std::vector<Complex> ReadFloatDiagonal(const float *data, int len) {
  std::vector<Complex> out;
  if (data == nullptr || len <= 0) {
    return out;
  }
  out.reserve(static_cast<std::size_t>(len));
  for (int i = 0; i < len; ++i) {
    out.emplace_back(static_cast<double>(data[i]), 0.0);
  }
  return out;
}

std::vector<Complex> ReadComplexDiagonal(const double *data, int len) {
  std::vector<Complex> out;
  if (data == nullptr || len <= 0) {
    return out;
  }
  const int count = len / 2;
  out.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    out.emplace_back(data[2 * i], data[2 * i + 1]);
  }
  return out;
}

int StorePlaintext(std::vector<Complex> values, int level, uint64_t scale) {
  const int id = AllocateId();
  g_scheme.plaintexts.emplace(
      id,
      TensorState{std::move(values), std::max(0, level), std::max<uint64_t>(1, scale), 0});
  return id;
}

int StoreCiphertext(std::vector<Complex> values, int level, uint64_t scale, int degree) {
  const int id = AllocateId();
  g_scheme.ciphertexts.emplace(
      id,
      TensorState{
          std::move(values),
          std::max(0, level),
          std::max<uint64_t>(1, scale),
          std::max(0, degree),
      });
  return id;
}

int StoreLinearTransform(std::vector<int> diag_indices,
                         std::vector<std::vector<Complex>> diagonals,
                         int level, int slots) {
  const int id = AllocateId();
  g_scheme.linear_transforms.emplace(
      id,
      LinearTransformState{
          std::move(diag_indices),
          std::move(diagonals),
          std::max(0, level),
          std::max(0, slots),
      });
  return id;
}

std::vector<Complex> RotateValues(const std::vector<Complex> &values, int amount) {
  const int n = static_cast<int>(values.size());
  std::vector<Complex> out(values.size(), Complex{0.0, 0.0});
  if (n == 0) {
    return out;
  }
  int shift = amount % n;
  if (shift < 0) {
    shift += n;
  }
  for (int j = 0; j < n; ++j) {
    int src = (j - shift) % n;
    if (src < 0) {
      src += n;
    }
    out[static_cast<std::size_t>(j)] = values[static_cast<std::size_t>(src)];
  }
  return out;
}

std::vector<Complex> BinaryValues(const std::vector<Complex> &lhs,
                                  const std::vector<Complex> &rhs,
                                  char op) {
  const std::size_t n = std::max(lhs.size(), rhs.size());
  std::vector<Complex> out(n, Complex{0.0, 0.0});
  for (std::size_t i = 0; i < n; ++i) {
    const Complex a = i < lhs.size() ? lhs[i] : Complex{0.0, 0.0};
    const Complex b = i < rhs.size() ? rhs[i] : Complex{0.0, 0.0};
    if (op == '+') {
      out[i] = a + b;
    } else if (op == '-') {
      out[i] = a - b;
    } else {
      out[i] = a * b;
    }
  }
  return out;
}

int BinaryCipherCipher(int lhs_id, int rhs_id, char op, bool in_place, bool relin) {
  TensorState &lhs = Ciphertext(lhs_id);
  TensorState &rhs = Ciphertext(rhs_id);
  std::vector<Complex> values = BinaryValues(lhs.values, rhs.values, op);
  const int level = std::min(lhs.level, rhs.level);
  const uint64_t scale = std::max(lhs.scale, rhs.scale);
  const int degree = relin ? 1 : std::max(lhs.degree, rhs.degree);
  if (in_place) {
    lhs.values = std::move(values);
    lhs.level = level;
    lhs.scale = scale;
    lhs.degree = degree;
    return lhs_id;
  }
  return StoreCiphertext(std::move(values), level, scale, degree);
}

int BinaryCipherPlain(int ciphertext_id, int plaintext_id, char op, bool in_place) {
  TensorState &lhs = Ciphertext(ciphertext_id);
  TensorState &rhs = Plaintext(plaintext_id);
  std::vector<Complex> values = BinaryValues(lhs.values, rhs.values, op);
  const int level = std::min(lhs.level, rhs.level);
  const uint64_t scale = std::max(lhs.scale, rhs.scale);
  if (in_place) {
    lhs.values = std::move(values);
    lhs.level = level;
    lhs.scale = scale;
    return ciphertext_id;
  }
  return StoreCiphertext(std::move(values), level, scale, lhs.degree);
}

int ScalarCipher(int ciphertext_id, Complex scalar, char op, bool in_place) {
  TensorState &state = Ciphertext(ciphertext_id);
  std::vector<Complex> values = state.values;
  for (Complex &value : values) {
    if (op == '+') {
      value += scalar;
    } else if (op == '-') {
      value -= scalar;
    } else {
      value *= scalar;
    }
  }
  if (in_place) {
    state.values = std::move(values);
    return ciphertext_id;
  }
  return StoreCiphertext(std::move(values), state.level, state.scale, state.degree);
}

std::vector<Complex> EvaluateLinearTransformValues(const LinearTransformState &transform,
                                                   const TensorState &input) {
  const int slots = transform.slots > 0 ? transform.slots : static_cast<int>(input.values.size());
  std::vector<Complex> output(static_cast<std::size_t>(slots), Complex{0.0, 0.0});
  if (slots <= 0 || input.values.empty()) {
    return output;
  }
  for (std::size_t d = 0; d < transform.diag_indices.size(); ++d) {
    const int diag_idx = transform.diag_indices[d];
    if (d >= transform.diagonals.size() || transform.diagonals[d].empty()) {
      continue;
    }
    const std::vector<Complex> &diag = transform.diagonals[d];
    for (int j = 0; j < slots; ++j) {
      int src = (j + diag_idx) % static_cast<int>(input.values.size());
      if (src < 0) {
        src += static_cast<int>(input.values.size());
      }
      const Complex coeff = diag[static_cast<std::size_t>(j) % diag.size()];
      output[static_cast<std::size_t>(j)] += coeff * input.values[static_cast<std::size_t>(src)];
    }
  }
  return output;
}

std::vector<int> UniqueSortedNonZeroKeys(const std::vector<int> &keys) {
  std::set<int> unique;
  for (int key : keys) {
    if (key != 0) {
      unique.insert(key);
    }
  }
  return std::vector<int>(unique.begin(), unique.end());
}

std::vector<int> RotationKeyRequestsFor(const std::vector<int> &keys, int level) {
  std::vector<int> flat;
  for (int key : UniqueSortedNonZeroKeys(keys)) {
    flat.push_back(key);
    flat.push_back(level);
  }
  return flat;
}

void AppendU64(std::vector<unsigned char> &out, uint64_t value) {
  for (int i = 0; i < 8; ++i) {
    out.push_back(static_cast<unsigned char>((value >> (8 * i)) & 0xffU));
  }
}

uint64_t ReadU64(const unsigned char *data, std::size_t len, std::size_t *cursor) {
  if (*cursor + 8 > len) {
    AbortWithMessage("clear_lattigo truncated byte payload");
  }
  uint64_t value = 0;
  for (int i = 0; i < 8; ++i) {
    value |= static_cast<uint64_t>(data[*cursor + i]) << (8 * i);
  }
  *cursor += 8;
  return value;
}

void AppendDouble(std::vector<unsigned char> &out, double value) {
  static_assert(sizeof(double) == 8, "double must be 8 bytes");
  unsigned char bytes[8];
  std::memcpy(bytes, &value, sizeof(double));
  out.insert(out.end(), bytes, bytes + sizeof(double));
}

double ReadDouble(const unsigned char *data, std::size_t len, std::size_t *cursor) {
  if (*cursor + sizeof(double) > len) {
    AbortWithMessage("clear_lattigo truncated double payload");
  }
  double value = 0.0;
  std::memcpy(&value, data + *cursor, sizeof(double));
  *cursor += sizeof(double);
  return value;
}

std::vector<unsigned char> SerializeVector(const std::vector<Complex> &values) {
  bool complex_payload = false;
  for (const Complex &value : values) {
    if (std::abs(value.imag()) > 0.0) {
      complex_payload = true;
      break;
    }
  }
  std::vector<unsigned char> out;
  AppendU64(out, static_cast<uint64_t>(values.size()));
  out.push_back(complex_payload ? 1 : 0);
  for (const Complex &value : values) {
    AppendDouble(out, value.real());
    if (complex_payload) {
      AppendDouble(out, value.imag());
    }
  }
  return out;
}

std::vector<Complex> DeserializeVector(const unsigned char *data, std::size_t len) {
  std::size_t cursor = 0;
  const uint64_t n = ReadU64(data, len, &cursor);
  if (cursor >= len) {
    AbortWithMessage("clear_lattigo missing vector kind byte");
  }
  const bool complex_payload = data[cursor++] != 0;
  std::vector<Complex> values;
  values.reserve(static_cast<std::size_t>(n));
  for (uint64_t i = 0; i < n; ++i) {
    const double real = ReadDouble(data, len, &cursor);
    const double imag = complex_payload ? ReadDouble(data, len, &cursor) : 0.0;
    values.emplace_back(real, imag);
  }
  return values;
}

int StorePolynomial(PolynomialState::Kind kind, const float *coeffs, int len_coeffs) {
  const int id = AllocateId();
  PolynomialState poly;
  poly.kind = kind;
  if (coeffs != nullptr && len_coeffs > 0) {
    poly.coeffs.reserve(static_cast<std::size_t>(len_coeffs));
    for (int i = 0; i < len_coeffs; ++i) {
      poly.coeffs.push_back(static_cast<double>(coeffs[i]));
    }
  }
  g_scheme.polynomials.emplace(id, std::move(poly));
  return id;
}

Complex EvaluateMonomial(const std::vector<double> &coeffs, Complex x) {
  Complex value{0.0, 0.0};
  for (double coeff : coeffs) {
    value = value * x + coeff;
  }
  return value;
}

Complex EvaluateChebyshev(const std::vector<double> &coeffs, Complex x) {
  if (coeffs.empty()) {
    return Complex{0.0, 0.0};
  }
  Complex t0{1.0, 0.0};
  Complex acc = coeffs[0] * t0;
  if (coeffs.size() == 1) {
    return acc;
  }
  Complex t1 = x;
  acc += coeffs[1] * t1;
  for (std::size_t k = 2; k < coeffs.size(); ++k) {
    Complex tk = Complex{2.0, 0.0} * x * t1 - t0;
    acc += coeffs[k] * tk;
    t0 = t1;
    t1 = tk;
  }
  return acc;
}

}  // namespace

extern "C" {

void FreeCArray(void *ptr) { std::free(ptr); }

void NewScheme(int logN, const int *logQ, int lenQ, const int *logP, int lenP,
               int logScale, int /*h*/, const char *ringType,
               const char * /*keysPath*/, const char * /*ioMode*/) {
  std::lock_guard<std::mutex> lock(g_mu);
  g_scheme = ClearScheme{};
  g_scheme.logn = static_cast<int>(logN);
  g_scheme.logq = ReadIntArray(logQ, lenQ);
  g_scheme.logp = ReadIntArray(logP, lenP);
  g_scheme.logscale = static_cast<int>(logScale);
  g_scheme.ring_type = ringType == nullptr ? "standard" : std::string(ringType);
  const int log_slots = g_scheme.ring_type == "standard" ? std::max(0, g_scheme.logn - 1)
                                                        : std::max(0, g_scheme.logn);
  g_scheme.slots = 1 << log_slots;
}

void DeleteScheme() {
  std::lock_guard<std::mutex> lock(g_mu);
  g_scheme = ClearScheme{};
}

ArrayResultUInt64 GetRuntimeMemoryStats() {
  return MakeUInt64ArrayResult(std::vector<unsigned long long>(12, 0));
}

void ResetOperationCounters() {
  std::lock_guard<std::mutex> lock(g_mu);
  g_scheme.rotations_total = 0;
  g_scheme.rotations_lt = 0;
  g_scheme.rotations_direct = 0;
  g_scheme.conjugations = 0;
}

ArrayResultUInt64 GetOperationCounters() {
  std::lock_guard<std::mutex> lock(g_mu);
  return MakeUInt64ArrayResult({
      static_cast<unsigned long long>(g_scheme.rotations_total),
      static_cast<unsigned long long>(g_scheme.rotations_lt),
      static_cast<unsigned long long>(g_scheme.rotations_direct),
      static_cast<unsigned long long>(g_scheme.conjugations),
  });
}

void DeletePlaintext(int plaintextID) {
  std::lock_guard<std::mutex> lock(g_mu);
  g_scheme.plaintexts.erase(plaintextID);
}

void DeleteCiphertext(int ciphertextID) {
  std::lock_guard<std::mutex> lock(g_mu);
  g_scheme.ciphertexts.erase(ciphertextID);
}

unsigned long GetPlaintextScale(int plaintextID) {
  std::lock_guard<std::mutex> lock(g_mu);
  return static_cast<unsigned long>(Plaintext(plaintextID).scale);
}

double GetPlaintextScaleLog2(int plaintextID) {
  std::lock_guard<std::mutex> lock(g_mu);
  return std::log2(static_cast<double>(Plaintext(plaintextID).scale));
}

unsigned long GetCiphertextScale(int ciphertextID) {
  std::lock_guard<std::mutex> lock(g_mu);
  return static_cast<unsigned long>(Ciphertext(ciphertextID).scale);
}

double GetCiphertextScaleLog2(int ciphertextID) {
  std::lock_guard<std::mutex> lock(g_mu);
  return std::log2(static_cast<double>(Ciphertext(ciphertextID).scale));
}

void SetPlaintextScale(int plaintextID, unsigned long scale) {
  std::lock_guard<std::mutex> lock(g_mu);
  Plaintext(plaintextID).scale = std::max<uint64_t>(1, static_cast<uint64_t>(scale));
}

void SetCiphertextScale(int ciphertextID, unsigned long scale) {
  std::lock_guard<std::mutex> lock(g_mu);
  Ciphertext(ciphertextID).scale = std::max<uint64_t>(1, static_cast<uint64_t>(scale));
}

int GetPlaintextLevel(int plaintextID) {
  std::lock_guard<std::mutex> lock(g_mu);
  return Plaintext(plaintextID).level;
}

int GetCiphertextLevel(int ciphertextID) {
  std::lock_guard<std::mutex> lock(g_mu);
  return Ciphertext(ciphertextID).level;
}

int GetPlaintextSlots(int plaintextID) {
  std::lock_guard<std::mutex> lock(g_mu);
  return static_cast<int>(Plaintext(plaintextID).values.size());
}

int GetCiphertextSlots(int ciphertextID) {
  std::lock_guard<std::mutex> lock(g_mu);
  return static_cast<int>(Ciphertext(ciphertextID).values.size());
}

int GetCiphertextDegree(int ciphertextID) {
  std::lock_guard<std::mutex> lock(g_mu);
  return Ciphertext(ciphertextID).degree;
}

ArrayResultUInt64 GetModuliChain() {
  std::lock_guard<std::mutex> lock(g_mu);
  std::vector<unsigned long long> values;
  for (int log_value : g_scheme.logq) {
    values.push_back(static_cast<unsigned long long>(ModulusProxyFromLog(log_value)));
  }
  return MakeUInt64ArrayResult(values);
}

ArrayResultUInt64 GetAuxModuliChain() {
  std::lock_guard<std::mutex> lock(g_mu);
  std::vector<unsigned long long> values;
  for (int log_value : g_scheme.logp) {
    values.push_back(static_cast<unsigned long long>(ModulusProxyFromLog(log_value)));
  }
  return MakeUInt64ArrayResult(values);
}

ArrayResultInt GetLivePlaintexts() {
  std::lock_guard<std::mutex> lock(g_mu);
  std::vector<int> ids;
  for (const auto &entry : g_scheme.plaintexts) {
    ids.push_back(entry.first);
  }
  return MakeIntArrayResult(ids);
}

ArrayResultInt GetLiveCiphertexts() {
  std::lock_guard<std::mutex> lock(g_mu);
  std::vector<int> ids;
  for (const auto &entry : g_scheme.ciphertexts) {
    ids.push_back(entry.first);
  }
  return MakeIntArrayResult(ids);
}

void NewKeyGenerator() {}
void GenerateSecretKey() {}
void GeneratePublicKey() {}
void GenerateRelinearizationKey() {}
void GenerateEvaluationKeys() {}

ArrayResultByte SerializeSecretKey() {
  return MakeByteArrayResult({0x63, 0x6c, 0x65, 0x61, 0x72});
}

void LoadSecretKey(const unsigned char * /*data*/, unsigned long /*lenData*/) {}

void NewEncoder() {}

int Encode(const float *values, int lenValues, int level, unsigned long scale) {
  std::lock_guard<std::mutex> lock(g_mu);
  std::vector<Complex> payload;
  if (values != nullptr && lenValues > 0) {
    payload.reserve(static_cast<std::size_t>(lenValues));
    for (int i = 0; i < lenValues; ++i) {
      payload.emplace_back(static_cast<double>(values[i]), 0.0);
    }
  }
  return StorePlaintext(std::move(payload), level, static_cast<uint64_t>(scale));
}

ArrayResultFloat Decode(int plaintextID) {
  std::lock_guard<std::mutex> lock(g_mu);
  const TensorState &state = Plaintext(plaintextID);
  std::vector<float> values;
  values.reserve(state.values.size());
  for (const Complex &value : state.values) {
    values.push_back(static_cast<float>(value.real()));
  }
  return MakeFloatArrayResult(values);
}

ArrayResultDouble DecodeComplex(int plaintextID) {
  std::lock_guard<std::mutex> lock(g_mu);
  const TensorState &state = Plaintext(plaintextID);
  std::vector<double> values;
  values.reserve(state.values.size() * 2);
  for (const Complex &value : state.values) {
    values.push_back(value.real());
    values.push_back(value.imag());
  }
  return MakeDoubleArrayResult(values);
}

void NewEncryptor() {}
void NewDecryptor() {}

int Encrypt(int plaintextID) {
  std::lock_guard<std::mutex> lock(g_mu);
  const TensorState state = Plaintext(plaintextID);
  return StoreCiphertext(state.values, state.level, state.scale, 1);
}

int Decrypt(int ciphertextID) {
  std::lock_guard<std::mutex> lock(g_mu);
  const TensorState state = Ciphertext(ciphertextID);
  return StorePlaintext(state.values, state.level, state.scale);
}

void NewEvaluator() {}

void AddRotationKey(int rotation) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (rotation != 0) {
    g_scheme.rotation_keys.insert(rotation);
  }
}

int Negate(int ciphertextID) {
  std::lock_guard<std::mutex> lock(g_mu);
  const TensorState &state = Ciphertext(ciphertextID);
  std::vector<Complex> values = state.values;
  for (Complex &value : values) {
    value = -value;
  }
  return StoreCiphertext(std::move(values), state.level, state.scale, state.degree);
}

int Conjugate(int ciphertextID) {
  std::lock_guard<std::mutex> lock(g_mu);
  TensorState &state = Ciphertext(ciphertextID);
  for (Complex &value : state.values) {
    value = std::conj(value);
  }
  g_scheme.conjugations += 1;
  return ciphertextID;
}

int ConjugateNew(int ciphertextID) {
  std::lock_guard<std::mutex> lock(g_mu);
  const TensorState &state = Ciphertext(ciphertextID);
  std::vector<Complex> values = state.values;
  for (Complex &value : values) {
    value = std::conj(value);
  }
  g_scheme.conjugations += 1;
  return StoreCiphertext(std::move(values), state.level, state.scale, state.degree);
}

int Rotate(int ciphertextID, int amount) {
  std::lock_guard<std::mutex> lock(g_mu);
  TensorState &state = Ciphertext(ciphertextID);
  state.values = RotateValues(state.values, amount);
  if (amount != 0) {
    g_scheme.rotation_keys.insert(amount);
    g_scheme.rotations_total += 1;
    g_scheme.rotations_direct += 1;
  }
  return ciphertextID;
}

int RotateNew(int ciphertextID, int amount) {
  std::lock_guard<std::mutex> lock(g_mu);
  const TensorState &state = Ciphertext(ciphertextID);
  if (amount != 0) {
    g_scheme.rotation_keys.insert(amount);
    g_scheme.rotations_total += 1;
    g_scheme.rotations_direct += 1;
  }
  return StoreCiphertext(RotateValues(state.values, amount), state.level, state.scale, state.degree);
}

int Rescale(int ciphertextID) {
  std::lock_guard<std::mutex> lock(g_mu);
  TensorState &state = Ciphertext(ciphertextID);
  state.level = std::max(0, state.level - 1);
  return ciphertextID;
}

int RescaleNew(int ciphertextID) {
  std::lock_guard<std::mutex> lock(g_mu);
  const TensorState &state = Ciphertext(ciphertextID);
  return StoreCiphertext(state.values, std::max(0, state.level - 1), state.scale, state.degree);
}

int AddScalar(int ciphertextID, float scalar) {
  std::lock_guard<std::mutex> lock(g_mu);
  return ScalarCipher(ciphertextID, Complex{scalar, 0.0}, '+', true);
}

int AddScalarNew(int ciphertextID, float scalar) {
  std::lock_guard<std::mutex> lock(g_mu);
  return ScalarCipher(ciphertextID, Complex{scalar, 0.0}, '+', false);
}

int SubScalar(int ciphertextID, float scalar) {
  std::lock_guard<std::mutex> lock(g_mu);
  return ScalarCipher(ciphertextID, Complex{scalar, 0.0}, '-', true);
}

int SubScalarNew(int ciphertextID, float scalar) {
  std::lock_guard<std::mutex> lock(g_mu);
  return ScalarCipher(ciphertextID, Complex{scalar, 0.0}, '-', false);
}

int MulScalarInt(int ciphertextID, int scalar) {
  std::lock_guard<std::mutex> lock(g_mu);
  return ScalarCipher(ciphertextID, Complex{static_cast<double>(scalar), 0.0}, '*', true);
}

int MulScalarIntNew(int ciphertextID, int scalar) {
  std::lock_guard<std::mutex> lock(g_mu);
  return ScalarCipher(ciphertextID, Complex{static_cast<double>(scalar), 0.0}, '*', false);
}

int MulScalarFloat(int ciphertextID, float scalar) {
  std::lock_guard<std::mutex> lock(g_mu);
  return ScalarCipher(ciphertextID, Complex{scalar, 0.0}, '*', true);
}

int MulScalarFloatNew(int ciphertextID, float scalar) {
  std::lock_guard<std::mutex> lock(g_mu);
  return ScalarCipher(ciphertextID, Complex{scalar, 0.0}, '*', false);
}

int MulImaginaryUnit(int ciphertextID, int sign) {
  std::lock_guard<std::mutex> lock(g_mu);
  return ScalarCipher(ciphertextID, Complex{0.0, static_cast<double>(sign)}, '*', true);
}

int MulImaginaryUnitNew(int ciphertextID, int sign) {
  std::lock_guard<std::mutex> lock(g_mu);
  return ScalarCipher(ciphertextID, Complex{0.0, static_cast<double>(sign)}, '*', false);
}

int AddPlaintext(int ciphertextID, int plaintextID) {
  std::lock_guard<std::mutex> lock(g_mu);
  return BinaryCipherPlain(ciphertextID, plaintextID, '+', true);
}

int AddPlaintextNew(int ciphertextID, int plaintextID) {
  std::lock_guard<std::mutex> lock(g_mu);
  return BinaryCipherPlain(ciphertextID, plaintextID, '+', false);
}

int SubPlaintext(int ciphertextID, int plaintextID) {
  std::lock_guard<std::mutex> lock(g_mu);
  return BinaryCipherPlain(ciphertextID, plaintextID, '-', true);
}

int SubPlaintextNew(int ciphertextID, int plaintextID) {
  std::lock_guard<std::mutex> lock(g_mu);
  return BinaryCipherPlain(ciphertextID, plaintextID, '-', false);
}

int MulPlaintext(int ciphertextID, int plaintextID) {
  std::lock_guard<std::mutex> lock(g_mu);
  return BinaryCipherPlain(ciphertextID, plaintextID, '*', true);
}

int MulPlaintextNew(int ciphertextID, int plaintextID) {
  std::lock_guard<std::mutex> lock(g_mu);
  return BinaryCipherPlain(ciphertextID, plaintextID, '*', false);
}

int AddCiphertext(int lhsID, int rhsID) {
  std::lock_guard<std::mutex> lock(g_mu);
  return BinaryCipherCipher(lhsID, rhsID, '+', true, false);
}

int AddCiphertextNew(int lhsID, int rhsID) {
  std::lock_guard<std::mutex> lock(g_mu);
  return BinaryCipherCipher(lhsID, rhsID, '+', false, false);
}

int SubCiphertext(int lhsID, int rhsID) {
  std::lock_guard<std::mutex> lock(g_mu);
  return BinaryCipherCipher(lhsID, rhsID, '-', true, false);
}

int SubCiphertextNew(int lhsID, int rhsID) {
  std::lock_guard<std::mutex> lock(g_mu);
  return BinaryCipherCipher(lhsID, rhsID, '-', false, false);
}

int MulRelinCiphertext(int lhsID, int rhsID) {
  std::lock_guard<std::mutex> lock(g_mu);
  return BinaryCipherCipher(lhsID, rhsID, '*', true, true);
}

int MulRelinCiphertextNew(int lhsID, int rhsID) {
  std::lock_guard<std::mutex> lock(g_mu);
  return BinaryCipherCipher(lhsID, rhsID, '*', false, true);
}

void NewPolynomialEvaluator() {}

int GenerateMonomial(const float *coeffs, int lenCoeffs) {
  std::lock_guard<std::mutex> lock(g_mu);
  return StorePolynomial(PolynomialState::Kind::Monomial, coeffs, lenCoeffs);
}

int GenerateChebyshev(const float *coeffs, int lenCoeffs) {
  std::lock_guard<std::mutex> lock(g_mu);
  return StorePolynomial(PolynomialState::Kind::Chebyshev, coeffs, lenCoeffs);
}

int EvaluatePolynomial(int ciphertextID, int polyID, unsigned long outScale) {
  std::lock_guard<std::mutex> lock(g_mu);
  const TensorState &state = Ciphertext(ciphertextID);
  const PolynomialState &poly = Polynomial(polyID);
  std::vector<Complex> values;
  values.reserve(state.values.size());
  for (const Complex &x : state.values) {
    values.push_back(
        poly.kind == PolynomialState::Kind::Chebyshev
            ? EvaluateChebyshev(poly.coeffs, x)
            : EvaluateMonomial(poly.coeffs, x));
  }
  return StoreCiphertext(std::move(values), state.level, static_cast<uint64_t>(outScale), state.degree);
}

ArrayResultDouble GenerateMinimaxSignCoeffs(const int *degrees, int lenDegrees,
                                            int /*prec*/, int /*logalpha*/,
                                            int /*logerr*/, int /*debug*/) {
  std::lock_guard<std::mutex> lock(g_mu);
  std::vector<double> coeffs;
  if (degrees != nullptr && lenDegrees > 0) {
    for (int i = 0; i < lenDegrees; ++i) {
      const int degree = std::max(0, degrees[i]);
      for (int j = 0; j <= degree; ++j) {
        coeffs.push_back(j == 1 ? 1.0 : 0.0);
      }
    }
  }
  return MakeDoubleArrayResult(coeffs);
}

int GetPolyDepth(int polyID) {
  std::lock_guard<std::mutex> lock(g_mu);
  const PolynomialState &poly = Polynomial(polyID);
  const int count = std::max(1, static_cast<int>(poly.coeffs.size()));
  return static_cast<int>(std::ceil(std::log2(static_cast<double>(count))));
}

void NewLinearTransformEvaluator() {}

int GenerateLinearTransform(const int *diagIdxs, int diagIdxsLen,
                            const float *diagData, int diagDataLen,
                            int level, float /*bsgsRatio*/, const char *ioMode) {
  std::lock_guard<std::mutex> lock(g_mu);
  const std::vector<int> indices = ReadIntArray(diagIdxs, diagIdxsLen);
  const bool load_mode = ioMode != nullptr && std::string(ioMode) == "load";
  const int slots = load_mode ? g_scheme.slots
                              : (diagIdxsLen > 0 ? diagDataLen / diagIdxsLen : g_scheme.slots);
  std::vector<std::vector<Complex>> diagonals;
  for (int i = 0; i < diagIdxsLen; ++i) {
    if (load_mode) {
      diagonals.emplace_back();
    } else {
      diagonals.push_back(ReadFloatDiagonal(diagData + i * slots, slots));
    }
  }
  return StoreLinearTransform(indices, std::move(diagonals), level, slots);
}

ArrayResultInt GenerateLinearTransformsBatch(int numTransforms,
                                             const int **diagIdxsArray,
                                             const int *diagIdxsLens,
                                             const float **diagDataArray,
                                             const int *diagDataLens,
                                             const int *levels,
                                             float bsgsRatio,
                                             const char *ioMode) {
  std::lock_guard<std::mutex> lock(g_mu);
  std::vector<int> ids;
  for (int i = 0; i < numTransforms; ++i) {
    const int diag_len = diagIdxsLens == nullptr ? 0 : diagIdxsLens[i];
    const int data_len = diagDataLens == nullptr ? 0 : diagDataLens[i];
    const std::vector<int> indices = ReadIntArray(diagIdxsArray == nullptr ? nullptr : diagIdxsArray[i], diag_len);
    const bool load_mode = ioMode != nullptr && std::string(ioMode) == "load";
    const int slots = load_mode ? g_scheme.slots : (diag_len > 0 ? data_len / diag_len : g_scheme.slots);
    std::vector<std::vector<Complex>> diagonals;
    for (int d = 0; d < diag_len; ++d) {
      if (load_mode) {
        diagonals.emplace_back();
      } else {
        diagonals.push_back(ReadFloatDiagonal(diagDataArray[i] + d * slots, slots));
      }
    }
    ids.push_back(StoreLinearTransform(indices, std::move(diagonals), levels == nullptr ? MaxLevel() : levels[i], slots));
  }
  (void)bsgsRatio;
  return MakeIntArrayResult(ids);
}

int EvaluateLinearTransform(int transformID, int ctxtID) {
  std::lock_guard<std::mutex> lock(g_mu);
  const LinearTransformState &transform = LinearTransform(transformID);
  const TensorState &input = Ciphertext(ctxtID);
  const int nonzero_keys = static_cast<int>(UniqueSortedNonZeroKeys(transform.diag_indices).size());
  g_scheme.rotations_total += static_cast<uint64_t>(nonzero_keys);
  g_scheme.rotations_lt += static_cast<uint64_t>(nonzero_keys);
  return StoreCiphertext(
      EvaluateLinearTransformValues(transform, input),
      std::min(input.level, transform.level),
      input.scale,
      input.degree);
}

void DeleteLinearTransform(int transformID) {
  std::lock_guard<std::mutex> lock(g_mu);
  g_scheme.linear_transforms.erase(transformID);
}

int GetLiveLinearTransformCount() {
  std::lock_guard<std::mutex> lock(g_mu);
  return static_cast<int>(g_scheme.linear_transforms.size());
}

ArrayResultInt GetLinearTransformRotationKeys(int transformID) {
  std::lock_guard<std::mutex> lock(g_mu);
  return MakeIntArrayResult(UniqueSortedNonZeroKeys(LinearTransform(transformID).diag_indices));
}

ArrayResultInt PlanLinearTransformRotationKeys(const int *diagIdxs, int diagIdxsLen,
                                               int /*level*/, float /*bsgsRatio*/) {
  std::lock_guard<std::mutex> lock(g_mu);
  return MakeIntArrayResult(UniqueSortedNonZeroKeys(ReadIntArray(diagIdxs, diagIdxsLen)));
}

ArrayResultInt PlanLinearTransformRotationKeyRequests(const int *diagIdxs, int diagIdxsLen,
                                                      int level, float /*bsgsRatio*/) {
  std::lock_guard<std::mutex> lock(g_mu);
  return MakeIntArrayResult(RotationKeyRequestsFor(ReadIntArray(diagIdxs, diagIdxsLen), level));
}

ArrayResultInt GetLinearTransformEmptyPlaintextKeys(int transformID) {
  std::lock_guard<std::mutex> lock(g_mu);
  const LinearTransformState &transform = LinearTransform(transformID);
  std::vector<int> empty;
  for (std::size_t i = 0; i < transform.diag_indices.size(); ++i) {
    if (i >= transform.diagonals.size() || transform.diagonals[i].empty()) {
      empty.push_back(transform.diag_indices[i]);
    }
  }
  return MakeIntArrayResult(empty);
}

void GenerateLinearTransformRotationKey(int key) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (key != 0) {
    g_scheme.rotation_keys.insert(key);
  }
}

ArrayResultInt GenerateLinearTransformsUnified(int numTransforms,
                                               const int **diagIdxsArray,
                                               const int *diagIdxsLens,
                                               const float **diagDataArray,
                                               const int *diagDataLens,
                                               const int *levels) {
  return GenerateLinearTransformsBatch(
      numTransforms, diagIdxsArray, diagIdxsLens, diagDataArray, diagDataLens, levels, 1.0f, "none");
}

ArrayResultInt PlanLinearTransformsUnifiedRotationKeys(int numTransforms,
                                                       const int **diagIdxsArray,
                                                       const int *diagIdxsLens,
                                                       const int * /*levels*/) {
  std::lock_guard<std::mutex> lock(g_mu);
  std::set<int> keys;
  for (int i = 0; i < numTransforms; ++i) {
    std::vector<int> indices = ReadIntArray(diagIdxsArray == nullptr ? nullptr : diagIdxsArray[i],
                                            diagIdxsLens == nullptr ? 0 : diagIdxsLens[i]);
    for (int key : indices) {
      if (key != 0) {
        keys.insert(key);
      }
    }
  }
  return MakeIntArrayResult(std::vector<int>(keys.begin(), keys.end()));
}

ArrayResultInt GenerateLinearTransformsUnifiedComplex(int numTransforms,
                                                      const int **diagIdxsArray,
                                                      const int *diagIdxsLens,
                                                      const double **diagDataArray,
                                                      const int *diagDataLens,
                                                      const int *levels) {
  std::lock_guard<std::mutex> lock(g_mu);
  std::vector<int> ids;
  for (int i = 0; i < numTransforms; ++i) {
    const int diag_len = diagIdxsLens == nullptr ? 0 : diagIdxsLens[i];
    const int data_len = diagDataLens == nullptr ? 0 : diagDataLens[i];
    const int slots = diag_len > 0 ? data_len / (2 * diag_len) : g_scheme.slots;
    std::vector<std::vector<Complex>> diagonals;
    for (int d = 0; d < diag_len; ++d) {
      diagonals.push_back(ReadComplexDiagonal(diagDataArray[i] + d * slots * 2, slots * 2));
    }
    ids.push_back(StoreLinearTransform(
        ReadIntArray(diagIdxsArray == nullptr ? nullptr : diagIdxsArray[i], diag_len),
        std::move(diagonals),
        levels == nullptr ? MaxLevel() : levels[i],
        slots));
  }
  return MakeIntArrayResult(ids);
}

ArrayResultInt GenerateLinearTransformsUnifiedLoad(int numTransforms,
                                                   const int **diagIdxsArray,
                                                   const int *diagIdxsLens,
                                                   const int *levels) {
  std::lock_guard<std::mutex> lock(g_mu);
  std::vector<int> ids;
  for (int i = 0; i < numTransforms; ++i) {
    const int diag_len = diagIdxsLens == nullptr ? 0 : diagIdxsLens[i];
    std::vector<std::vector<Complex>> diagonals(static_cast<std::size_t>(std::max(0, diag_len)));
    ids.push_back(StoreLinearTransform(
        ReadIntArray(diagIdxsArray == nullptr ? nullptr : diagIdxsArray[i], diag_len),
        std::move(diagonals),
        levels == nullptr ? MaxLevel() : levels[i],
        g_scheme.slots));
  }
  return MakeIntArrayResult(ids);
}

ArrayResultInt EvaluateLinearTransformsWithSharedCache(const int *transformIDs,
                                                       int numTransforms,
                                                       int ctxtID) {
  std::lock_guard<std::mutex> lock(g_mu);
  const TensorState input = Ciphertext(ctxtID);
  std::vector<int> out_ids;
  for (int i = 0; i < numTransforms; ++i) {
    const LinearTransformState &transform = LinearTransform(transformIDs[i]);
    const int nonzero_keys = static_cast<int>(UniqueSortedNonZeroKeys(transform.diag_indices).size());
    g_scheme.rotations_total += static_cast<uint64_t>(nonzero_keys);
    g_scheme.rotations_lt += static_cast<uint64_t>(nonzero_keys);
    out_ids.push_back(StoreCiphertext(
        EvaluateLinearTransformValues(transform, input),
        std::min(input.level, transform.level),
        input.scale,
        input.degree));
  }
  return MakeIntArrayResult(out_ids);
}

ArrayResultInt EvaluateLinearTransformSourcesWithSharedCacheAdd(
    const int *ctxtIDs, int numSources, const int *transformIDs,
    const int *targetIDs, const int *groupOffsets, int numPartials,
    int numTargets) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (numTargets <= 0) {
    return MakeIntArrayResult({});
  }
  std::vector<std::vector<Complex>> outputs(static_cast<std::size_t>(numTargets));
  std::vector<int> levels(static_cast<std::size_t>(numTargets), MaxLevel());
  std::vector<uint64_t> scales(static_cast<std::size_t>(numTargets), ModulusProxyFromLog(g_scheme.logscale));
  std::vector<int> degrees(static_cast<std::size_t>(numTargets), 1);
  for (int source = 0; source < numSources; ++source) {
    const TensorState input = Ciphertext(ctxtIDs[source]);
    const int start = groupOffsets[source];
    const int end = groupOffsets[source + 1];
    for (int partial = start; partial < end && partial < numPartials; ++partial) {
      const int target = targetIDs[partial];
      const LinearTransformState &transform = LinearTransform(transformIDs[partial]);
      std::vector<Complex> values = EvaluateLinearTransformValues(transform, input);
      const int nonzero_keys = static_cast<int>(UniqueSortedNonZeroKeys(transform.diag_indices).size());
      g_scheme.rotations_total += static_cast<uint64_t>(nonzero_keys);
      g_scheme.rotations_lt += static_cast<uint64_t>(nonzero_keys);
      if (target < 0 || target >= numTargets) {
        continue;
      }
      if (outputs[static_cast<std::size_t>(target)].empty()) {
        outputs[static_cast<std::size_t>(target)] = std::move(values);
      } else {
        outputs[static_cast<std::size_t>(target)] =
            BinaryValues(outputs[static_cast<std::size_t>(target)], values, '+');
      }
      levels[static_cast<std::size_t>(target)] =
          std::min(levels[static_cast<std::size_t>(target)], std::min(input.level, transform.level));
      scales[static_cast<std::size_t>(target)] = input.scale;
      degrees[static_cast<std::size_t>(target)] = input.degree;
    }
  }
  std::vector<int> out_ids;
  for (int target = 0; target < numTargets; ++target) {
    out_ids.push_back(StoreCiphertext(
        std::move(outputs[static_cast<std::size_t>(target)]),
        levels[static_cast<std::size_t>(target)],
        scales[static_cast<std::size_t>(target)],
        degrees[static_cast<std::size_t>(target)]));
  }
  return MakeIntArrayResult(out_ids);
}

void EnableLinearTransformEvaluationProfile(int /*enabled*/) {}
void ResetLinearTransformEvaluationProfile() {}

ArrayResultUInt64 GetLinearTransformEvaluationProfileCounters() {
  return MakeUInt64ArrayResult(std::vector<unsigned long long>(8, 0));
}

ArrayResultDouble GetLinearTransformEvaluationProfileSeconds() {
  return MakeDoubleArrayResult(std::vector<double>(8, 0.0));
}

ArrayResultDouble ConsumeSharedCacheEvalProfileSeconds() {
  return MakeDoubleArrayResult(std::vector<double>(11, 0.0));
}

int LinearTransformUsesStreaming(int /*transformID*/) { return 0; }

ArrayResultByte GenerateAndSerializeRotationKey(int key) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (key != 0) {
    g_scheme.rotation_keys.insert(key);
  }
  std::vector<unsigned char> bytes;
  AppendU64(bytes, static_cast<uint64_t>(static_cast<int64_t>(key)));
  return MakeByteArrayResult(bytes);
}

void LoadRotationKey(const unsigned char * /*data*/, unsigned long /*lenData*/, unsigned long key) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (key != 0) {
    g_scheme.rotation_keys.insert(static_cast<int>(key));
  }
}

ArrayResultByte SerializeDiagonal(int transformID, int diagIdx) {
  std::lock_guard<std::mutex> lock(g_mu);
  const LinearTransformState &transform = LinearTransform(transformID);
  for (std::size_t i = 0; i < transform.diag_indices.size(); ++i) {
    if (transform.diag_indices[i] == diagIdx && i < transform.diagonals.size()) {
      return MakeByteArrayResult(SerializeVector(transform.diagonals[i]));
    }
  }
  return MakeByteArrayResult({});
}

void LoadPlaintextDiagonal(const unsigned char *data, unsigned long lenData,
                           int transformID, unsigned long diagIdx) {
  std::lock_guard<std::mutex> lock(g_mu);
  LinearTransformState &transform = LinearTransform(transformID);
  std::vector<Complex> values = DeserializeVector(data, lenData);
  for (std::size_t i = 0; i < transform.diag_indices.size(); ++i) {
    if (transform.diag_indices[i] == static_cast<int>(diagIdx)) {
      if (i >= transform.diagonals.size()) {
        transform.diagonals.resize(i + 1);
      }
      transform.diagonals[i] = std::move(values);
      return;
    }
  }
  transform.diag_indices.push_back(static_cast<int>(diagIdx));
  transform.diagonals.push_back(std::move(values));
}

void LoadPlaintextDiagonalsBatch(const unsigned char *payload, unsigned long lenPayload,
                                 const unsigned long long *offsets, int lenOffsets,
                                 const unsigned long long *lengths, int lenLengths,
                                 const int *diagIndices, int lenDiagIndices,
                                 int transformID) {
  std::lock_guard<std::mutex> lock(g_mu);
  LinearTransformState &transform = LinearTransform(transformID);
  const int count = std::min({lenOffsets, lenLengths, lenDiagIndices});
  for (int i = 0; i < count; ++i) {
    const std::size_t offset = static_cast<std::size_t>(offsets[i]);
    const std::size_t length = static_cast<std::size_t>(lengths[i]);
    if (payload == nullptr || offset + length > static_cast<std::size_t>(lenPayload)) {
      continue;
    }
    std::vector<Complex> values = DeserializeVector(payload + offset, length);
    bool installed = false;
    for (std::size_t j = 0; j < transform.diag_indices.size(); ++j) {
      if (transform.diag_indices[j] == diagIndices[i]) {
        if (j >= transform.diagonals.size()) {
          transform.diagonals.resize(j + 1);
        }
        transform.diagonals[j] = values;
        installed = true;
        break;
      }
    }
    if (!installed) {
      transform.diag_indices.push_back(diagIndices[i]);
      transform.diagonals.push_back(std::move(values));
    }
  }
}

void RemovePlaintextDiagonals(int transformID) {
  std::lock_guard<std::mutex> lock(g_mu);
  LinearTransformState &transform = LinearTransform(transformID);
  for (std::vector<Complex> &diag : transform.diagonals) {
    diag.clear();
  }
}

ArrayResultInt GetLinearTransformPlaintextLevels(int transformID) {
  std::lock_guard<std::mutex> lock(g_mu);
  const LinearTransformState &transform = LinearTransform(transformID);
  return MakeIntArrayResult(std::vector<int>(transform.diag_indices.size(), transform.level));
}

void RemoveRotationKeys() {
  std::lock_guard<std::mutex> lock(g_mu);
  g_scheme.rotation_keys.clear();
}

void NewBootstrapper(const int * /*logPs*/, int /*lenLogPs*/, int /*numSlots*/) {}

int Bootstrap(int ciphertextID, int /*numSlots*/) {
  std::lock_guard<std::mutex> lock(g_mu);
  const TensorState &state = Ciphertext(ciphertextID);
  return StoreCiphertext(state.values, MaxLevel(), state.scale, state.degree);
}

ArrayResultInt BootstrapMany(const int *ciphertextIDs, int lenCiphertextIDs, int numSlots) {
  std::lock_guard<std::mutex> lock(g_mu);
  std::vector<int> out;
  for (int i = 0; i < lenCiphertextIDs; ++i) {
    const TensorState &state = Ciphertext(ciphertextIDs[i]);
    out.push_back(StoreCiphertext(state.values, MaxLevel(), state.scale, state.degree));
  }
  (void)numSlots;
  return MakeIntArrayResult(out);
}

void EnableBootstrapProfile(int /*enabled*/) {}
void ResetBootstrapProfile() {}

ArrayResultUInt64 GetBootstrapProfileCounters() {
  return MakeUInt64ArrayResult(std::vector<unsigned long long>(8, 0));
}

void DeleteBootstrappers() {}

}  // extern "C"
