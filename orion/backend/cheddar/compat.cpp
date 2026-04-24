#include "compat.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "UserInterface.h"
#include "common/Assert.h"
#include "core/Container.h"
#include "core/Context.h"
#include "extension/BootContext.h"
#include "extension/BootParameter.h"
#include "extension/EvalPoly.h"
#include "extension/Hoist.h"
#include "extension/LinearTransform.h"
#include "extension/StripedMatrix.h"

namespace {

using word = uint64_t;
using Complex = cheddar::Complex;
using Ct = cheddar::Ciphertext<word>;
using Pt = cheddar::Plaintext<word>;
using Const = cheddar::Constant<word>;
using Parameter = cheddar::Parameter<word>;
using BootContext = cheddar::BootContext<word>;
using HoistHandler = cheddar::HoistHandler<word>;
using LinearTransform = cheddar::LinearTransform<word>;
using EvalPoly = cheddar::EvalPoly<word>;
using UserInterface = cheddar::UserInterface<word>;

template <typename T>
class HeapAllocator {
 public:
  int Add(T &&obj) {
    const int id = AllocateId();
    entries_.emplace(id, std::make_unique<T>(std::move(obj)));
    return id;
  }

  T &Get(int id) {
    auto it = entries_.find(id);
    if (it == entries_.end()) {
      throw std::runtime_error("Heap object not found");
    }
    return *it->second;
  }

  const T &Get(int id) const {
    auto it = entries_.find(id);
    if (it == entries_.end()) {
      throw std::runtime_error("Heap object not found");
    }
    return *it->second;
  }

  void Delete(int id) {
    auto it = entries_.find(id);
    if (it == entries_.end()) {
      return;
    }
    entries_.erase(it);
    freed_.push(id);
  }

  void Reset() {
    entries_.clear();
    freed_ = std::priority_queue<int, std::vector<int>, std::greater<int>>();
    next_id_ = 0;
  }

  std::vector<int> LiveKeys() const {
    std::vector<int> keys;
    keys.reserve(entries_.size());
    for (const auto &[key, _] : entries_) {
      keys.push_back(key);
    }
    return keys;
  }

 private:
  int AllocateId() {
    if (!freed_.empty()) {
      const int id = freed_.top();
      freed_.pop();
      return id;
    }
    return next_id_++;
  }

  int next_id_ = 0;
  std::priority_queue<int, std::vector<int>, std::greater<int>> freed_;
  std::map<int, std::unique_ptr<T>> entries_;
};

struct PolynomialSpec {
  std::vector<double> coeffs;
  bool chebyshev = false;
};

struct LinearTransformState {
  std::unique_ptr<LinearTransform> transform;
  cheddar::StripedMatrix matrix;
  std::vector<int> diag_indices;
  int level = 0;
  int bs = 2;
  int gs = 1;

  LinearTransformState(std::unique_ptr<LinearTransform> value,
                       cheddar::StripedMatrix raw_matrix,
                       std::vector<int> indices, int transform_level,
                       int baby_step, int giant_step)
      : transform(std::move(value)),
        matrix(std::move(raw_matrix)),
        diag_indices(std::move(indices)),
        level(transform_level),
        bs(baby_step),
        gs(giant_step) {}
};

struct SharedCacheBucket {
  int level = 0;
  std::vector<int> transform_ids;
  std::unique_ptr<HoistHandler> cache;
};

struct SharedCachePlan {
  std::vector<SharedCacheBucket> buckets;
};

struct SchemeState {
  std::unique_ptr<Parameter> param;
  std::shared_ptr<BootContext> context;
  std::unique_ptr<UserInterface> interface;
  HeapAllocator<Pt> plaintexts;
  HeapAllocator<Ct> ciphertexts;
  HeapAllocator<PolynomialSpec> polynomials;
  HeapAllocator<LinearTransformState> transforms;
  std::map<std::vector<int>, SharedCachePlan> shared_cache_plans;
  bool eval_mod_prepared = false;
  std::set<int> prepared_boot_slots;
};

std::unique_ptr<SchemeState> g_scheme;

[[noreturn]] void AbortWithMessage(const char *message) {
  std::fprintf(stderr, "Cheddar backend error: %s\n", message);
  std::abort();
}

void RequireScheme() {
  if (!g_scheme) {
    AbortWithMessage("scheme is not initialized");
  }
}

void EnsureRotationKeyPrepared(int key);

template <typename T>
T *AllocArray(std::size_t count) {
  if (count == 0) {
    return nullptr;
  }
  void *ptr = std::malloc(sizeof(T) * count);
  if (ptr == nullptr) {
    AbortWithMessage("malloc failed");
  }
  return static_cast<T *>(ptr);
}

template <typename T>
ArrayResultInt MakeIntArrayResult(const std::vector<T> &values) {
  ArrayResultInt result{nullptr, 0};
  if (values.empty()) {
    return result;
  }
  result.Data = AllocArray<int>(values.size());
  result.Length = static_cast<unsigned long>(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    result.Data[i] = static_cast<int>(values[i]);
  }
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

ArrayResultUInt64 MakeUInt64ArrayResult(
    const std::vector<unsigned long long> &values) {
  ArrayResultUInt64 result{nullptr, 0};
  if (values.empty()) {
    return result;
  }
  result.Data = AllocArray<unsigned long long>(values.size());
  result.Length = static_cast<unsigned long>(values.size());
  std::memcpy(result.Data, values.data(),
              sizeof(unsigned long long) * values.size());
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

struct Preset40_64 {
  int log_degree = 16;
  int log_default_scale = 40;
  int default_encryption_level = 13;
  int num_cts_levels = 4;
  int num_stc_levels = 3;
  std::vector<word> terminal_primes{};
  std::vector<word> main_primes{
      1125899908022273ULL, 1099515691009ULL, 1099523555329ULL,
      1099525128193ULL,    1099526176769ULL, 1099529060353ULL,
      1099535220737ULL,    1099536138241ULL, 1099537580033ULL,
      1099538104321ULL,    1099540725761ULL, 1099540856833ULL,
      1099543085057ULL,    36028797019488257ULL, 36028797023420417ULL,
      36028797024206849ULL, 36028797025124353ULL, 36028797032202241ULL,
      36028797033644033ULL, 36028797037576193ULL, 36028797048324097ULL,
      36028797048586241ULL, 36028797049896961ULL, 36028797051863041ULL,
      36028797053698049ULL, 36028797054222337ULL,
  };
  std::vector<word> auxiliary_primes{
      72057594038321153ULL, 72057594040680449ULL, 72057594042646529ULL,
      72057594047889409ULL, 72057594057195521ULL, 72057594058375169ULL,
      72057594058899457ULL,
  };
  std::vector<std::pair<int, int>> level_config{
      {1, 0},  {2, 0},  {3, 0},  {4, 0},  {5, 0},  {6, 0},  {7, 0},
      {8, 0},  {9, 0},  {10, 0}, {11, 0}, {12, 0}, {13, 0}, {14, 0},
      {15, 0}, {16, 0}, {17, 0}, {18, 0}, {19, 0}, {20, 0}, {21, 0},
      {22, 0}, {23, 0}, {24, 0}, {25, 0}, {26, 0},
  };
  std::pair<int, int> additional_base{0, 0};
};

std::unique_ptr<SchemeState> BuildPreset40Scheme() {
  Preset40_64 preset;
  auto state = std::make_unique<SchemeState>();
  state->param = std::make_unique<Parameter>(
      preset.log_degree, static_cast<double>(1ULL << preset.log_default_scale),
      preset.default_encryption_level, preset.level_config, preset.main_primes,
      preset.auxiliary_primes, preset.terminal_primes, preset.additional_base);
  const cheddar::BootParameter boot_param(state->param->max_level_,
                                          preset.num_cts_levels,
                                          preset.num_stc_levels);
  state->context = BootContext::Create(*state->param, boot_param);
  state->interface = std::make_unique<UserInterface>(state->context);
  return state;
}

int MaxRequestedPrimeBits(const int *values, int length) {
  int max_bits = 0;
  for (int i = 0; i < length; ++i) {
    max_bits = std::max(max_bits, values[i]);
  }
  return max_bits;
}

int PushPlaintext(Pt &&plaintext) {
  return g_scheme->plaintexts.Add(std::move(plaintext));
}

int PushCiphertext(Ct &&ciphertext) {
  return g_scheme->ciphertexts.Add(std::move(ciphertext));
}

Pt &RetrievePlaintext(int id) { return g_scheme->plaintexts.Get(id); }

Ct &RetrieveCiphertext(int id) { return g_scheme->ciphertexts.Get(id); }

PolynomialSpec &RetrievePolynomial(int id) {
  return g_scheme->polynomials.Get(id);
}

LinearTransformState &RetrieveTransform(int id) {
  return g_scheme->transforms.Get(id);
}

LinearTransform &EnsureTransformLoaded(LinearTransformState &state) {
  if (!state.transform) {
    state.transform = std::make_unique<LinearTransform>(
        g_scheme->context, state.matrix, state.level,
        g_scheme->param->GetScale(state.level), state.bs, state.gs);
  }
  return *state.transform;
}

int CiphertextDegree(const Ct &ciphertext) { return ciphertext.HasRx() ? 2 : 1; }

int CiphertextLevel(const Ct &ciphertext) {
  return g_scheme->param->NPToLevel(ciphertext.GetNP());
}

int PlaintextLevel(const Pt &plaintext) {
  return g_scheme->param->NPToLevel(plaintext.GetNP());
}

std::vector<Complex> DecodeMessage(const Pt &plaintext) {
  std::vector<Complex> decoded;
  g_scheme->context->encoder_.Decode(decoded, plaintext);
  return decoded;
}

Pt EncodeMessage(const std::vector<Complex> &values, int level, double scale) {
  Pt plaintext;
  g_scheme->context->encoder_.Encode(plaintext, level, scale, values);
  return plaintext;
}

Const EncodeScalarConstant(int level, double scale, double value) {
  Const constant;
  g_scheme->context->encoder_.EncodeConstant(constant, level, scale, value);
  return constant;
}

std::vector<float> ExtractRealComponents(const std::vector<Complex> &values) {
  std::vector<float> out(values.size(), 0.0f);
  for (std::size_t i = 0; i < values.size(); ++i) {
    out[i] = static_cast<float>(values[i].real());
  }
  return out;
}

std::vector<double> FlattenComplexInterleaved(
    const std::vector<Complex> &values) {
  std::vector<double> flat;
  flat.reserve(values.size() * 2);
  for (const Complex &value : values) {
    flat.push_back(value.real());
    flat.push_back(value.imag());
  }
  return flat;
}

template <typename T>
void AppendPod(std::vector<unsigned char> &buffer, const T &value) {
  const unsigned char *raw =
      reinterpret_cast<const unsigned char *>(&value);
  buffer.insert(buffer.end(), raw, raw + sizeof(T));
}

template <typename T>
T ReadPod(const unsigned char *&cursor, const unsigned char *end) {
  if (static_cast<std::size_t>(end - cursor) < sizeof(T)) {
    AbortWithMessage("serialized payload is truncated");
  }
  T value;
  std::memcpy(&value, cursor, sizeof(T));
  cursor += sizeof(T);
  return value;
}

std::vector<unsigned char> SerializeEvaluationKeyBytes(
    const cheddar::EvaluationKey<word> &key) {
  const cheddar::NPInfo np = key.GetNP();
  const int beta = key.GetBeta();
  const std::uint64_t vec_size =
      beta == 0 ? 0 : static_cast<std::uint64_t>(key.bx_.at(0).size());

  std::vector<unsigned char> payload;
  AppendPod(payload, static_cast<std::int32_t>(np.num_main_));
  AppendPod(payload, static_cast<std::int32_t>(np.num_ter_));
  AppendPod(payload, static_cast<std::int32_t>(np.num_aux_));
  AppendPod(payload, static_cast<std::int32_t>(beta));
  AppendPod(payload, vec_size);

  for (int index = 0; index < beta; ++index) {
    cheddar::HostVector<word> bx_host;
    cheddar::HostVector<word> ax_host;
    cheddar::CopyDeviceToHost(bx_host, key.bx_.at(index));
    cheddar::CopyDeviceToHost(ax_host, key.ax_.at(index));
    const unsigned char *bx_raw =
        reinterpret_cast<const unsigned char *>(bx_host.data());
    const unsigned char *ax_raw =
        reinterpret_cast<const unsigned char *>(ax_host.data());
    payload.insert(payload.end(), bx_raw,
                   bx_raw + bx_host.size() * sizeof(word));
    payload.insert(payload.end(), ax_raw,
                   ax_raw + ax_host.size() * sizeof(word));
  }
  return payload;
}

cheddar::EvaluationKey<word> DeserializeEvaluationKeyBytes(
    const unsigned char *data, std::size_t size) {
  const unsigned char *cursor = data;
  const unsigned char *end = data + size;

  const int num_main = ReadPod<std::int32_t>(cursor, end);
  const int num_ter = ReadPod<std::int32_t>(cursor, end);
  const int num_aux = ReadPod<std::int32_t>(cursor, end);
  const int beta = ReadPod<std::int32_t>(cursor, end);
  const std::uint64_t vec_size = ReadPod<std::uint64_t>(cursor, end);

  cheddar::EvaluationKey<word> key(cheddar::NPInfo(num_main, num_ter, num_aux),
                                   beta);
  for (int index = 0; index < beta; ++index) {
    cheddar::HostVector<word> bx_host(static_cast<std::size_t>(vec_size));
    cheddar::HostVector<word> ax_host(static_cast<std::size_t>(vec_size));
    const std::size_t byte_count = static_cast<std::size_t>(vec_size) * sizeof(word);
    if (static_cast<std::size_t>(end - cursor) < byte_count * 2) {
      AbortWithMessage("serialized evaluation key payload is truncated");
    }
    std::memcpy(bx_host.data(), cursor, byte_count);
    cursor += byte_count;
    std::memcpy(ax_host.data(), cursor, byte_count);
    cursor += byte_count;
    cheddar::CopyHostToDevice(key.bx_.at(index), bx_host);
    cheddar::CopyHostToDevice(key.ax_.at(index), ax_host);
  }
  return key;
}

std::vector<unsigned char> SerializeSecretBytes() {
  cheddar::HostVector<word> main_secret;
  cheddar::HostVector<word> sparse_secret;
  g_scheme->interface->ExportSecrets(main_secret, sparse_secret);

  std::vector<unsigned char> payload;
  const std::uint64_t main_size = static_cast<std::uint64_t>(main_secret.size());
  const std::uint64_t sparse_size =
      static_cast<std::uint64_t>(sparse_secret.size());
  AppendPod(payload, main_size);
  AppendPod(payload, sparse_size);
  const unsigned char *main_raw =
      reinterpret_cast<const unsigned char *>(main_secret.data());
  const unsigned char *sparse_raw =
      reinterpret_cast<const unsigned char *>(sparse_secret.data());
  payload.insert(payload.end(), main_raw,
                 main_raw + main_secret.size() * sizeof(word));
  payload.insert(payload.end(), sparse_raw,
                 sparse_raw + sparse_secret.size() * sizeof(word));
  return payload;
}

void LoadSecretBytes(const unsigned char *data, std::size_t size) {
  const unsigned char *cursor = data;
  const unsigned char *end = data + size;
  const std::uint64_t main_size = ReadPod<std::uint64_t>(cursor, end);
  const std::uint64_t sparse_size = ReadPod<std::uint64_t>(cursor, end);
  const std::size_t main_bytes = static_cast<std::size_t>(main_size) * sizeof(word);
  const std::size_t sparse_bytes =
      static_cast<std::size_t>(sparse_size) * sizeof(word);
  if (static_cast<std::size_t>(end - cursor) < main_bytes + sparse_bytes) {
    AbortWithMessage("serialized secret payload is truncated");
  }

  cheddar::HostVector<word> main_secret(static_cast<std::size_t>(main_size));
  cheddar::HostVector<word> sparse_secret(static_cast<std::size_t>(sparse_size));
  std::memcpy(main_secret.data(), cursor, main_bytes);
  cursor += main_bytes;
  std::memcpy(sparse_secret.data(), cursor, sparse_bytes);
  g_scheme->interface->LoadSecrets(main_secret, sparse_secret);
}

std::vector<unsigned char> SerializeDiagonalBytes(
    const LinearTransformState &state, int diag_idx) {
  auto it = state.matrix.find(diag_idx);
  if (it == state.matrix.end()) {
    AbortWithMessage("requested diagonal does not exist");
  }
  const std::vector<Complex> &diag = it->second;
  std::vector<unsigned char> payload;
  AppendPod(payload, static_cast<std::uint64_t>(diag.size()));
  for (const Complex &value : diag) {
    AppendPod(payload, value.real());
    AppendPod(payload, value.imag());
  }
  return payload;
}

template <typename Fn>
int ApplyCiphertextUnaryInPlace(int ciphertext_id, Fn &&fn) {
  Ct result;
  fn(result, RetrieveCiphertext(ciphertext_id));
  RetrieveCiphertext(ciphertext_id) = std::move(result);
  return ciphertext_id;
}

template <typename Fn>
int ApplyCiphertextUnaryNew(int ciphertext_id, Fn &&fn) {
  Ct result;
  fn(result, RetrieveCiphertext(ciphertext_id));
  return PushCiphertext(std::move(result));
}

template <typename Fn>
int ApplyCiphertextBinaryInPlace(int lhs_id, int rhs_id, Fn &&fn) {
  Ct result;
  fn(result, RetrieveCiphertext(lhs_id), RetrieveCiphertext(rhs_id));
  RetrieveCiphertext(lhs_id) = std::move(result);
  return lhs_id;
}

template <typename Fn>
int ApplyCiphertextBinaryNew(int lhs_id, int rhs_id, Fn &&fn) {
  Ct result;
  fn(result, RetrieveCiphertext(lhs_id), RetrieveCiphertext(rhs_id));
  return PushCiphertext(std::move(result));
}

template <typename Fn>
int ApplyCiphertextPlainBinaryInPlace(int ciphertext_id, int plaintext_id,
                                      Fn &&fn) {
  Ct result;
  fn(result, RetrieveCiphertext(ciphertext_id), RetrievePlaintext(plaintext_id));
  RetrieveCiphertext(ciphertext_id) = std::move(result);
  return ciphertext_id;
}

template <typename Fn>
int ApplyCiphertextPlainBinaryNew(int ciphertext_id, int plaintext_id,
                                  Fn &&fn) {
  Ct result;
  fn(result, RetrieveCiphertext(ciphertext_id), RetrievePlaintext(plaintext_id));
  return PushCiphertext(std::move(result));
}

template <typename Fn>
int ApplyCiphertextConstInPlace(int ciphertext_id, double scalar,
                                double constant_scale, Fn &&fn) {
  Ct result;
  const Ct &ciphertext = RetrieveCiphertext(ciphertext_id);
  const int level = CiphertextLevel(ciphertext);
  Const constant = EncodeScalarConstant(level, constant_scale, scalar);
  fn(result, ciphertext, constant);
  RetrieveCiphertext(ciphertext_id) = std::move(result);
  return ciphertext_id;
}

template <typename Fn>
int ApplyCiphertextConstNew(int ciphertext_id, double scalar,
                            double constant_scale, Fn &&fn) {
  Ct result;
  const Ct &ciphertext = RetrieveCiphertext(ciphertext_id);
  const int level = CiphertextLevel(ciphertext);
  Const constant = EncodeScalarConstant(level, constant_scale, scalar);
  fn(result, ciphertext, constant);
  return PushCiphertext(std::move(result));
}

cheddar::StripedMatrix BuildRealStripedMatrix(const int *diag_idxs,
                                              int diag_count,
                                              const float *diag_data,
                                              int diag_data_len) {
  if (diag_count <= 0) {
    throw std::runtime_error("linear transform requires at least one diagonal");
  }
  const int slots = diag_data_len / diag_count;
  if (slots * diag_count != diag_data_len) {
    throw std::runtime_error("linear transform diagonal data has invalid size");
  }
  cheddar::StripedMatrix matrix(slots, slots);
  for (int diag_index = 0; diag_index < diag_count; ++diag_index) {
    std::vector<Complex> values(slots, Complex(0.0, 0.0));
    for (int slot = 0; slot < slots; ++slot) {
      values[slot] =
          Complex(static_cast<double>(diag_data[diag_index * slots + slot]), 0.0);
    }
    matrix[diag_idxs[diag_index]] = std::move(values);
  }
  return matrix;
}

cheddar::StripedMatrix BuildComplexStripedMatrix(const int *diag_idxs,
                                                 int diag_count,
                                                 const double *diag_data,
                                                 int diag_data_len) {
  if (diag_count <= 0) {
    throw std::runtime_error("linear transform requires at least one diagonal");
  }
  const int slots = diag_data_len / (diag_count * 2);
  if (slots * diag_count * 2 != diag_data_len) {
    throw std::runtime_error(
        "complex linear transform diagonal data has invalid size");
  }
  cheddar::StripedMatrix matrix(slots, slots);
  for (int diag_index = 0; diag_index < diag_count; ++diag_index) {
    std::vector<Complex> values(slots, Complex(0.0, 0.0));
    for (int slot = 0; slot < slots; ++slot) {
      const int offset = diag_index * slots * 2 + slot * 2;
      values[slot] = Complex(diag_data[offset], diag_data[offset + 1]);
    }
    matrix[diag_idxs[diag_index]] = std::move(values);
  }
  return matrix;
}

int AddLinearTransformFromMatrix(const cheddar::StripedMatrix &matrix,
                                 int level) {
  if (level < 0 || level > g_scheme->param->default_encryption_level_) {
    throw std::runtime_error(
        "linear transform level is outside supported preset range");
  }
  const int bs = std::max(2, matrix.GetNumDiag());
  auto transform = std::make_unique<LinearTransform>(
      g_scheme->context, matrix, level, g_scheme->param->GetScale(level), bs,
      1);
  std::vector<int> diag_indices;
  diag_indices.reserve(matrix.size());
  for (const auto &[diag_idx, _] : matrix) {
    diag_indices.push_back(diag_idx);
  }
  return g_scheme->transforms.Add(
      LinearTransformState(std::move(transform), matrix,
                           std::move(diag_indices), level, bs, 1));
}

std::unique_ptr<HoistHandler> BuildSharedCacheHoist(
    int level, const std::set<int> &baby_step_indices) {
  cheddar::PlainHoistMap hoist_map;
  auto &pt_map = hoist_map[0];
  const std::vector<Complex> dummy_message(2, Complex(0.0, 0.0));
  for (int bs_idx : baby_step_indices) {
    pt_map.emplace(bs_idx, dummy_message);
  }
  return std::make_unique<HoistHandler>(
      g_scheme->context, hoist_map, level, g_scheme->param->GetScale(level),
      true);
}

SharedCachePlan BuildSharedCachePlan(const std::vector<int> &ordered_ids) {
  SharedCachePlan plan;
  std::map<int, std::size_t> bucket_index_by_level;
  std::map<int, std::set<int>> bucket_bs_union;

  for (int transform_id : ordered_ids) {
    LinearTransformState &state = RetrieveTransform(transform_id);
    auto [it, inserted] =
        bucket_index_by_level.emplace(state.level, plan.buckets.size());
    if (inserted) {
      SharedCacheBucket bucket;
      bucket.level = state.level;
      plan.buckets.push_back(std::move(bucket));
    }
    SharedCacheBucket &bucket = plan.buckets[it->second];
    bucket.transform_ids.push_back(transform_id);
    std::set<int> &union_indices = bucket_bs_union[state.level];
    union_indices.insert(0);
    LinearTransform &transform = EnsureTransformLoaded(state);
    for (int bs_idx : transform.GetBabyStepIndices()) {
      union_indices.insert(bs_idx);
    }
    cheddar::EvkRequest req;
    transform.AddRequiredRotations(req);
    for (const auto &[rot_idx, _] : req) {
      if (rot_idx != 0) {
        EnsureRotationKeyPrepared(rot_idx);
      }
    }
  }

  for (SharedCacheBucket &bucket : plan.buckets) {
    bucket.cache = BuildSharedCacheHoist(bucket.level, bucket_bs_union[bucket.level]);
  }
  return plan;
}

SharedCachePlan &GetOrBuildSharedCachePlan(const std::vector<int> &ordered_ids) {
  auto it = g_scheme->shared_cache_plans.find(ordered_ids);
  if (it != g_scheme->shared_cache_plans.end()) {
    return it->second;
  }
  auto [inserted_it, _] = g_scheme->shared_cache_plans.emplace(
      ordered_ids, BuildSharedCachePlan(ordered_ids));
  return inserted_it->second;
}

void EnsureRotationKeyPrepared(int key) {
  if (key <= 0) {
    return;
  }
  g_scheme->interface->PrepareRotationKey(key, g_scheme->param->max_level_);
}

void EnsureBootstrapPrepared(int num_slots) {
  if (!g_scheme->eval_mod_prepared) {
    g_scheme->context->PrepareEvalMod();
    g_scheme->eval_mod_prepared = true;
  }
  if (g_scheme->prepared_boot_slots.insert(num_slots).second) {
    g_scheme->context->PrepareEvalSpecialFFT(num_slots);
    cheddar::EvkRequest req;
    g_scheme->context->AddRequiredRotations(req, num_slots);
    g_scheme->interface->PrepareRotationKey(req);
  }
}

ArrayResultDouble UnsupportedMinimax() {
  AbortWithMessage(
      "GenerateMinimaxSignCoeffs is not implemented in the C++ adapter yet");
}

}  // namespace

extern "C" {

void NewScheme(int logN, const int *logQ, int lenQ, const int *logP, int lenP,
               int logScale, int /*h*/, const char *ringType,
               const char * /*keysPath*/, const char *ioMode) {
  DeleteScheme();
  if (ringType == nullptr || std::strcmp(ringType, "standard") != 0) {
    AbortWithMessage("only RingType=standard is supported");
  }
  (void)ioMode;
  if (logN != 16) {
    AbortWithMessage(
        "only LogN=16 is supported in the first C++ adapter pass");
  }
  if (logScale != 40) {
    AbortWithMessage(
        "only LogScale=40 is supported in the first C++ adapter pass");
  }
  const int max_bits = std::max(MaxRequestedPrimeBits(logQ, lenQ),
                                MaxRequestedPrimeBits(logP, lenP));
  if (max_bits > 63) {
    AbortWithMessage("requested modulus sizes exceed uint64 support");
  }
  g_scheme = BuildPreset40Scheme();
}

void DeleteScheme() { g_scheme.reset(); }

void FreeCArray(void *ptr) { std::free(ptr); }

void NewKeyGenerator() { RequireScheme(); }
void GenerateSecretKey() { RequireScheme(); }
void GeneratePublicKey() { RequireScheme(); }
void GenerateRelinearizationKey() { RequireScheme(); }
void GenerateEvaluationKeys() { RequireScheme(); }

ArrayResultByte SerializeSecretKey() {
  RequireScheme();
  return MakeByteArrayResult(SerializeSecretBytes());
}

void LoadSecretKey(const unsigned char *data, unsigned long lenData) {
  RequireScheme();
  LoadSecretBytes(data, static_cast<std::size_t>(lenData));
}

void NewEncoder() { RequireScheme(); }

int Encode(const float *values, int lenValues, int level, unsigned long scale) {
  RequireScheme();
  std::vector<Complex> message(lenValues, Complex(0.0, 0.0));
  for (int i = 0; i < lenValues; ++i) {
    message[i] = Complex(static_cast<double>(values[i]), 0.0);
  }
  Pt plaintext = EncodeMessage(message, level, static_cast<double>(scale));
  return PushPlaintext(std::move(plaintext));
}

ArrayResultFloat Decode(int plaintextID) {
  RequireScheme();
  const Pt &plaintext = RetrievePlaintext(plaintextID);
  return MakeFloatArrayResult(ExtractRealComponents(DecodeMessage(plaintext)));
}

ArrayResultDouble DecodeComplex(int plaintextID) {
  RequireScheme();
  const Pt &plaintext = RetrievePlaintext(plaintextID);
  return MakeDoubleArrayResult(
      FlattenComplexInterleaved(DecodeMessage(plaintext)));
}

void NewEncryptor() { RequireScheme(); }
void NewDecryptor() { RequireScheme(); }

int Encrypt(int plaintextID) {
  RequireScheme();
  Ct ciphertext;
  g_scheme->interface->Encrypt(ciphertext, RetrievePlaintext(plaintextID));
  return PushCiphertext(std::move(ciphertext));
}

int Decrypt(int ciphertextID) {
  RequireScheme();
  Pt plaintext;
  g_scheme->interface->Decrypt(plaintext, RetrieveCiphertext(ciphertextID));
  return PushPlaintext(std::move(plaintext));
}

void DeletePlaintext(int plaintextID) {
  if (!g_scheme) {
    return;
  }
  g_scheme->plaintexts.Delete(plaintextID);
}

void DeleteCiphertext(int ciphertextID) {
  if (!g_scheme) {
    return;
  }
  g_scheme->ciphertexts.Delete(ciphertextID);
}

unsigned long GetPlaintextScale(int plaintextID) {
  RequireScheme();
  return static_cast<unsigned long>(RetrievePlaintext(plaintextID).GetScale());
}

double GetPlaintextScaleLog2(int plaintextID) {
  RequireScheme();
  return std::log2(RetrievePlaintext(plaintextID).GetScale());
}

unsigned long GetCiphertextScale(int ciphertextID) {
  RequireScheme();
  return static_cast<unsigned long>(RetrieveCiphertext(ciphertextID).GetScale());
}

double GetCiphertextScaleLog2(int ciphertextID) {
  RequireScheme();
  return std::log2(RetrieveCiphertext(ciphertextID).GetScale());
}

void SetPlaintextScale(int plaintextID, unsigned long scale) {
  RequireScheme();
  RetrievePlaintext(plaintextID).SetScale(static_cast<double>(scale));
}

void SetCiphertextScale(int ciphertextID, unsigned long scale) {
  RequireScheme();
  RetrieveCiphertext(ciphertextID).SetScale(static_cast<double>(scale));
}

int GetPlaintextLevel(int plaintextID) {
  RequireScheme();
  return PlaintextLevel(RetrievePlaintext(plaintextID));
}

int GetCiphertextLevel(int ciphertextID) {
  RequireScheme();
  return CiphertextLevel(RetrieveCiphertext(ciphertextID));
}

int GetPlaintextSlots(int plaintextID) {
  RequireScheme();
  return RetrievePlaintext(plaintextID).GetNumSlots();
}

int GetCiphertextSlots(int ciphertextID) {
  RequireScheme();
  return RetrieveCiphertext(ciphertextID).GetNumSlots();
}

int GetCiphertextDegree(int ciphertextID) {
  RequireScheme();
  return CiphertextDegree(RetrieveCiphertext(ciphertextID));
}

ArrayResultUInt64 GetModuliChain() {
  RequireScheme();
  const auto np = g_scheme->param->LevelToNP(g_scheme->param->max_level_);
  const std::vector<word> primes = g_scheme->param->GetPrimeVector(np);
  std::vector<unsigned long long> q_chain;
  q_chain.reserve(np.GetNumQ());
  for (int i = 0; i < np.GetNumQ(); ++i) {
    q_chain.push_back(static_cast<unsigned long long>(primes[i]));
  }
  return MakeUInt64ArrayResult(q_chain);
}

ArrayResultUInt64 GetAuxModuliChain() {
  RequireScheme();
  std::vector<unsigned long long> aux;
  aux.reserve(g_scheme->param->aux_primes_.size());
  for (word prime : g_scheme->param->aux_primes_) {
    aux.push_back(static_cast<unsigned long long>(prime));
  }
  return MakeUInt64ArrayResult(aux);
}

ArrayResultInt GetLivePlaintexts() {
  RequireScheme();
  return MakeIntArrayResult(g_scheme->plaintexts.LiveKeys());
}

ArrayResultInt GetLiveCiphertexts() {
  RequireScheme();
  return MakeIntArrayResult(g_scheme->ciphertexts.LiveKeys());
}

void NewEvaluator() { RequireScheme(); }

void AddRotationKey(int rotation) {
  RequireScheme();
  if (rotation < 0) {
    AbortWithMessage(
        "negative rotation keys are not supported in the first C++ adapter pass");
  }
  EnsureRotationKeyPrepared(rotation);
}

int Negate(int ciphertextID) {
  RequireScheme();
  return ApplyCiphertextUnaryNew(ciphertextID, [&](Ct &out, const Ct &in) {
    g_scheme->context->Neg(out, in);
  });
}

int Conjugate(int ciphertextID) {
  RequireScheme();
  return ApplyCiphertextUnaryInPlace(ciphertextID, [&](Ct &out, const Ct &in) {
    g_scheme->context->HConj(out, in,
                             g_scheme->interface->GetConjugationKey());
  });
}

int ConjugateNew(int ciphertextID) {
  RequireScheme();
  return ApplyCiphertextUnaryNew(ciphertextID, [&](Ct &out, const Ct &in) {
    g_scheme->context->HConj(out, in,
                             g_scheme->interface->GetConjugationKey());
  });
}

int Rotate(int ciphertextID, int amount) {
  RequireScheme();
  if (amount < 0) {
    AbortWithMessage(
        "negative rotations are not supported in the first C++ adapter pass");
  }
  EnsureRotationKeyPrepared(amount);
  return ApplyCiphertextUnaryInPlace(ciphertextID, [&](Ct &out, const Ct &in) {
    g_scheme->context->HRot(out, in,
                            g_scheme->interface->GetRotationKey(amount),
                            amount);
  });
}

int RotateNew(int ciphertextID, int amount) {
  RequireScheme();
  if (amount < 0) {
    AbortWithMessage(
        "negative rotations are not supported in the first C++ adapter pass");
  }
  EnsureRotationKeyPrepared(amount);
  return ApplyCiphertextUnaryNew(ciphertextID, [&](Ct &out, const Ct &in) {
    g_scheme->context->HRot(out, in,
                            g_scheme->interface->GetRotationKey(amount),
                            amount);
  });
}

int Rescale(int ciphertextID) {
  RequireScheme();
  return ApplyCiphertextUnaryInPlace(ciphertextID, [&](Ct &out, const Ct &in) {
    g_scheme->context->Rescale(out, in);
  });
}

int RescaleNew(int ciphertextID) {
  RequireScheme();
  return ApplyCiphertextUnaryNew(ciphertextID, [&](Ct &out, const Ct &in) {
    g_scheme->context->Rescale(out, in);
  });
}

int AddScalar(int ciphertextID, float scalar) {
  RequireScheme();
  return ApplyCiphertextConstInPlace(
      ciphertextID, static_cast<double>(scalar), 1.0,
      [&](Ct &out, const Ct &in, const Const &constant) {
        g_scheme->context->Add(out, in, constant);
      });
}

int AddScalarNew(int ciphertextID, float scalar) {
  RequireScheme();
  return ApplyCiphertextConstNew(
      ciphertextID, static_cast<double>(scalar), 1.0,
      [&](Ct &out, const Ct &in, const Const &constant) {
        g_scheme->context->Add(out, in, constant);
      });
}

int SubScalar(int ciphertextID, float scalar) {
  RequireScheme();
  return ApplyCiphertextConstInPlace(
      ciphertextID, static_cast<double>(scalar), 1.0,
      [&](Ct &out, const Ct &in, const Const &constant) {
        g_scheme->context->Sub(out, in, constant);
      });
}

int SubScalarNew(int ciphertextID, float scalar) {
  RequireScheme();
  return ApplyCiphertextConstNew(
      ciphertextID, static_cast<double>(scalar), 1.0,
      [&](Ct &out, const Ct &in, const Const &constant) {
        g_scheme->context->Sub(out, in, constant);
      });
}

int MulScalarInt(int ciphertextID, int scalar) {
  RequireScheme();
  return ApplyCiphertextConstInPlace(
      ciphertextID, static_cast<double>(scalar), 1.0,
      [&](Ct &out, const Ct &in, const Const &constant) {
        g_scheme->context->Mult(out, in, constant);
      });
}

int MulScalarIntNew(int ciphertextID, int scalar) {
  RequireScheme();
  return ApplyCiphertextConstNew(
      ciphertextID, static_cast<double>(scalar), 1.0,
      [&](Ct &out, const Ct &in, const Const &constant) {
        g_scheme->context->Mult(out, in, constant);
      });
}

int MulScalarFloat(int ciphertextID, float scalar) {
  RequireScheme();
  const double scale = RetrieveCiphertext(ciphertextID).GetScale();
  return ApplyCiphertextConstInPlace(
      ciphertextID, static_cast<double>(scalar), scale,
      [&](Ct &out, const Ct &in, const Const &constant) {
        g_scheme->context->Mult(out, in, constant);
      });
}

int MulScalarFloatNew(int ciphertextID, float scalar) {
  RequireScheme();
  const double scale = RetrieveCiphertext(ciphertextID).GetScale();
  return ApplyCiphertextConstNew(
      ciphertextID, static_cast<double>(scalar), scale,
      [&](Ct &out, const Ct &in, const Const &constant) {
        g_scheme->context->Mult(out, in, constant);
      });
}

int MulImaginaryUnit(int ciphertextID, int sign) {
  RequireScheme();
  return ApplyCiphertextUnaryInPlace(ciphertextID, [&](Ct &out, const Ct &in) {
    g_scheme->context->MultImaginaryUnit(out, in);
    if (sign < 0) {
      Ct negated;
      g_scheme->context->Neg(negated, out);
      out = std::move(negated);
    }
  });
}

int MulImaginaryUnitNew(int ciphertextID, int sign) {
  RequireScheme();
  return ApplyCiphertextUnaryNew(ciphertextID, [&](Ct &out, const Ct &in) {
    g_scheme->context->MultImaginaryUnit(out, in);
    if (sign < 0) {
      Ct negated;
      g_scheme->context->Neg(negated, out);
      out = std::move(negated);
    }
  });
}

int AddPlaintext(int ciphertextID, int plaintextID) {
  RequireScheme();
  return ApplyCiphertextPlainBinaryInPlace(
      ciphertextID, plaintextID, [&](Ct &out, const Ct &ct, const Pt &pt) {
        g_scheme->context->Add(out, ct, pt);
      });
}

int AddPlaintextNew(int ciphertextID, int plaintextID) {
  RequireScheme();
  return ApplyCiphertextPlainBinaryNew(
      ciphertextID, plaintextID, [&](Ct &out, const Ct &ct, const Pt &pt) {
        g_scheme->context->Add(out, ct, pt);
      });
}

int SubPlaintext(int ciphertextID, int plaintextID) {
  RequireScheme();
  return ApplyCiphertextPlainBinaryInPlace(
      ciphertextID, plaintextID, [&](Ct &out, const Ct &ct, const Pt &pt) {
        g_scheme->context->Sub(out, ct, pt);
      });
}

int SubPlaintextNew(int ciphertextID, int plaintextID) {
  RequireScheme();
  return ApplyCiphertextPlainBinaryNew(
      ciphertextID, plaintextID, [&](Ct &out, const Ct &ct, const Pt &pt) {
        g_scheme->context->Sub(out, ct, pt);
      });
}

int MulPlaintext(int ciphertextID, int plaintextID) {
  RequireScheme();
  return ApplyCiphertextPlainBinaryInPlace(
      ciphertextID, plaintextID, [&](Ct &out, const Ct &ct, const Pt &pt) {
        g_scheme->context->Mult(out, ct, pt);
      });
}

int MulPlaintextNew(int ciphertextID, int plaintextID) {
  RequireScheme();
  return ApplyCiphertextPlainBinaryNew(
      ciphertextID, plaintextID, [&](Ct &out, const Ct &ct, const Pt &pt) {
        g_scheme->context->Mult(out, ct, pt);
      });
}

int AddCiphertext(int lhsID, int rhsID) {
  RequireScheme();
  return ApplyCiphertextBinaryInPlace(
      lhsID, rhsID, [&](Ct &out, const Ct &lhs, const Ct &rhs) {
        g_scheme->context->Add(out, lhs, rhs);
      });
}

int AddCiphertextNew(int lhsID, int rhsID) {
  RequireScheme();
  return ApplyCiphertextBinaryNew(
      lhsID, rhsID, [&](Ct &out, const Ct &lhs, const Ct &rhs) {
        g_scheme->context->Add(out, lhs, rhs);
      });
}

int SubCiphertext(int lhsID, int rhsID) {
  RequireScheme();
  return ApplyCiphertextBinaryInPlace(
      lhsID, rhsID, [&](Ct &out, const Ct &lhs, const Ct &rhs) {
        g_scheme->context->Sub(out, lhs, rhs);
      });
}

int SubCiphertextNew(int lhsID, int rhsID) {
  RequireScheme();
  return ApplyCiphertextBinaryNew(
      lhsID, rhsID, [&](Ct &out, const Ct &lhs, const Ct &rhs) {
        g_scheme->context->Sub(out, lhs, rhs);
      });
}

int MulRelinCiphertext(int lhsID, int rhsID) {
  RequireScheme();
  return ApplyCiphertextBinaryInPlace(
      lhsID, rhsID, [&](Ct &out, const Ct &lhs, const Ct &rhs) {
        g_scheme->context->HMult(out, lhs, rhs,
                                 g_scheme->interface->GetMultiplicationKey(),
                                 false);
      });
}

int MulRelinCiphertextNew(int lhsID, int rhsID) {
  RequireScheme();
  return ApplyCiphertextBinaryNew(
      lhsID, rhsID, [&](Ct &out, const Ct &lhs, const Ct &rhs) {
        g_scheme->context->HMult(out, lhs, rhs,
                                 g_scheme->interface->GetMultiplicationKey(),
                                 false);
      });
}

void NewPolynomialEvaluator() { RequireScheme(); }

int GenerateMonomial(const float *coeffs, int lenCoeffs) {
  RequireScheme();
  PolynomialSpec spec;
  spec.coeffs.reserve(lenCoeffs);
  for (int i = lenCoeffs - 1; i >= 0; --i) {
    spec.coeffs.push_back(static_cast<double>(coeffs[i]));
  }
  spec.chebyshev = false;
  return g_scheme->polynomials.Add(std::move(spec));
}

int GenerateChebyshev(const float *coeffs, int lenCoeffs) {
  RequireScheme();
  PolynomialSpec spec;
  spec.coeffs.reserve(lenCoeffs);
  for (int i = 0; i < lenCoeffs; ++i) {
    spec.coeffs.push_back(static_cast<double>(coeffs[i]));
  }
  spec.chebyshev = true;
  return g_scheme->polynomials.Add(std::move(spec));
}

int EvaluatePolynomial(int ciphertextID, int polyID, unsigned long outScale) {
  RequireScheme();
  const PolynomialSpec &spec = RetrievePolynomial(polyID);
  const Ct &input = RetrieveCiphertext(ciphertextID);
  if (spec.coeffs.size() < 3) {
    AbortWithMessage(
        "degree < 2 polynomials are not supported in the first C++ adapter pass");
  }
  EvalPoly poly(spec.coeffs, CiphertextLevel(input), input.GetScale(),
                static_cast<double>(outScale), spec.chebyshev);
  poly.Compile(g_scheme->context);
  Ct output;
  poly.Evaluate(g_scheme->context, output, input,
                g_scheme->interface->GetMultiplicationKey());
  return PushCiphertext(std::move(output));
}

ArrayResultDouble GenerateMinimaxSignCoeffs(const int * /*degrees*/,
                                            int /*lenDegrees*/, int /*prec*/,
                                            int /*logalpha*/, int /*logerr*/,
                                            int /*debug*/) {
  RequireScheme();
  return UnsupportedMinimax();
}

int GetPolyDepth(int polyID) {
  RequireScheme();
  const PolynomialSpec &spec = RetrievePolynomial(polyID);
  if (spec.coeffs.empty()) {
    return 0;
  }
  return static_cast<int>(
      std::ceil(std::log2(static_cast<double>(spec.coeffs.size()))));
}

void NewLinearTransformEvaluator() { RequireScheme(); }

int GenerateLinearTransform(const int *diagIdxs, int diagIdxsLen,
                            const float *diagData, int diagDataLen, int level,
                            float /*bsgsRatio*/, const char * /*ioMode*/) {
  RequireScheme();
  return AddLinearTransformFromMatrix(
      BuildRealStripedMatrix(diagIdxs, diagIdxsLen, diagData, diagDataLen),
      level);
}

int EvaluateLinearTransform(int transformID, int ciphertextID) {
  RequireScheme();
  Ct output;
  LinearTransform &transform = EnsureTransformLoaded(RetrieveTransform(transformID));
  transform.Evaluate(g_scheme->context, output, RetrieveCiphertext(ciphertextID),
                     g_scheme->interface->GetEvkMap());
  return PushCiphertext(std::move(output));
}

void DeleteLinearTransform(int transformID) {
  if (!g_scheme) {
    return;
  }
  g_scheme->shared_cache_plans.clear();
  g_scheme->transforms.Delete(transformID);
}

ArrayResultInt GetLinearTransformRotationKeys(int transformID) {
  RequireScheme();
  cheddar::EvkRequest req;
  LinearTransform &transform = EnsureTransformLoaded(RetrieveTransform(transformID));
  transform.AddRequiredRotations(req);
  std::vector<int> keys;
  keys.reserve(req.size());
  for (const auto &[key, _] : req) {
    if (key != 0) {
      keys.push_back(key);
    }
  }
  return MakeIntArrayResult(keys);
}

void GenerateLinearTransformRotationKey(int key) {
  RequireScheme();
  EnsureRotationKeyPrepared(key);
}

ArrayResultInt GenerateLinearTransformsUnified(
    int numTransforms, const int *const *diagIdxsArray, const int *diagIdxsLens,
    const float *const *diagDataArray, const int *diagDataLens,
    const int *levels) {
  RequireScheme();
  std::vector<int> ids;
  ids.reserve(numTransforms);
  for (int i = 0; i < numTransforms; ++i) {
    ids.push_back(AddLinearTransformFromMatrix(
        BuildRealStripedMatrix(diagIdxsArray[i], diagIdxsLens[i],
                               diagDataArray[i], diagDataLens[i]),
        levels[i]));
  }
  return MakeIntArrayResult(ids);
}

ArrayResultInt GenerateLinearTransformsUnifiedComplex(
    int numTransforms, const int *const *diagIdxsArray, const int *diagIdxsLens,
    const double *const *diagDataArray, const int *diagDataLens,
    const int *levels) {
  RequireScheme();
  std::vector<int> ids;
  ids.reserve(numTransforms);
  for (int i = 0; i < numTransforms; ++i) {
    ids.push_back(AddLinearTransformFromMatrix(
        BuildComplexStripedMatrix(diagIdxsArray[i], diagIdxsLens[i],
                                  diagDataArray[i], diagDataLens[i]),
        levels[i]));
  }
  return MakeIntArrayResult(ids);
}

ArrayResultInt EvaluateLinearTransformsWithSharedCache(const int *transformIDs,
                                                       int numTransforms,
                                                       int ciphertextID) {
  RequireScheme();
  std::vector<int> ordered_ids;
  ordered_ids.reserve(numTransforms);
  for (int i = 0; i < numTransforms; ++i) {
    ordered_ids.push_back(transformIDs[i]);
  }
  SharedCachePlan &plan = GetOrBuildSharedCachePlan(ordered_ids);
  const Ct &input = RetrieveCiphertext(ciphertextID);
  const int input_level = CiphertextLevel(input);
  std::map<int, int> output_id_by_transform;

  for (const SharedCacheBucket &bucket : plan.buckets) {
    const Ct *bucket_input = &input;
    Ct leveled_input;
    if (input_level > bucket.level) {
      g_scheme->context->LevelDown(leveled_input, input, bucket.level);
      bucket_input = &leveled_input;
    } else if (input_level < bucket.level) {
      AbortWithMessage(
          "input ciphertext level is below the shared-cache transform level");
    }

    std::map<int, Ct> bs_cache;
    bucket.cache->EvaluateBabyStep(g_scheme->context, bs_cache, *bucket_input,
                                   g_scheme->interface->GetEvkMap(), false);

    for (int transform_id : bucket.transform_ids) {
      Ct output;
      LinearTransform &transform = EnsureTransformLoaded(RetrieveTransform(transform_id));
      transform.EvaluateGiantStep(g_scheme->context, output, bs_cache,
                                  g_scheme->interface->GetEvkMap(), false);
      output_id_by_transform[transform_id] = PushCiphertext(std::move(output));
    }
  }

  std::vector<int> output_ids;
  output_ids.reserve(numTransforms);
  for (int i = 0; i < numTransforms; ++i) {
    output_ids.push_back(output_id_by_transform.at(transformIDs[i]));
  }
  return MakeIntArrayResult(output_ids);
}

ArrayResultByte GenerateAndSerializeRotationKey(int key) {
  RequireScheme();
  EnsureRotationKeyPrepared(key);
  const auto bytes =
      SerializeEvaluationKeyBytes(g_scheme->interface->GetRotationKey(key));
  g_scheme->interface->RemoveRotationKeys();
  return MakeByteArrayResult(bytes);
}

void LoadRotationKey(const unsigned char *data, unsigned long lenData,
                     unsigned long key) {
  RequireScheme();
  auto evk = DeserializeEvaluationKeyBytes(data, static_cast<std::size_t>(lenData));
  g_scheme->interface->SetRotationKey(static_cast<int>(key), std::move(evk));
}

ArrayResultByte SerializeDiagonal(int transformID, int diagIdx) {
  RequireScheme();
  LinearTransformState &state = RetrieveTransform(transformID);
  const auto bytes = SerializeDiagonalBytes(state, diagIdx);
  state.transform.reset();
  g_scheme->shared_cache_plans.clear();
  return MakeByteArrayResult(bytes);
}

void LoadPlaintextDiagonal(const unsigned char * /*data*/,
                          unsigned long /*lenData*/, int transformID,
                          unsigned long /*diagIdx*/) {
  RequireScheme();
  EnsureTransformLoaded(RetrieveTransform(transformID));
}

void LoadPlaintextDiagonalsBatch(const unsigned char * /*data*/,
                                 unsigned long /*lenData*/,
                                 const unsigned long long * /*offsets*/,
                                 int /*numOffsets*/,
                                 const unsigned long long * /*lengths*/,
                                 int /*numLengths*/,
                                  const int * /*diagIdxs*/,
                                  int /*numDiagIdxs*/,
                                 int transformID) {
  RequireScheme();
  EnsureTransformLoaded(RetrieveTransform(transformID));
}

void RemovePlaintextDiagonals(int transformID) {
  RequireScheme();
  RetrieveTransform(transformID).transform.reset();
}

void RemoveRotationKeys() {
  RequireScheme();
  g_scheme->interface->RemoveRotationKeys();
}

void NewBootstrapper(const int * /*logPs*/, int /*lenLogPs*/, int numSlots) {
  RequireScheme();
  EnsureBootstrapPrepared(numSlots);
}

int Bootstrap(int ciphertextID, int numSlots) {
  RequireScheme();
  EnsureBootstrapPrepared(numSlots);
  Ct output;
  g_scheme->context->Boot(output, RetrieveCiphertext(ciphertextID),
                          g_scheme->interface->GetEvkMap());
  return PushCiphertext(std::move(output));
}

void DeleteBootstrappers() {
  if (!g_scheme) {
    return;
  }
  g_scheme->prepared_boot_slots.clear();
}

}  // extern "C"
