#include "compat.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <map>
#include <memory>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

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
using Evk = cheddar::EvaluationKey<word>;
using Parameter = cheddar::Parameter<word>;
using BootContext = cheddar::BootContext<word>;
using HoistHandler = cheddar::HoistHandler<word>;
using LinearTransform = cheddar::LinearTransform<word>;
using EvalPoly = cheddar::EvalPoly<word>;
using UserInterface = cheddar::UserInterface<word>;
using EvkMap = cheddar::EvkMap<word>;

constexpr bool kUseMinKSLinearTransforms = false;
double g_device_memory_trim_seconds = 0.0;

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
  std::unique_ptr<EvkMap> rotation_keys;
  cheddar::StripedMatrix matrix;
  std::vector<int> diag_indices;
  struct StreamingPlaintextPayloadChunk {
    std::vector<int> giant_steps;
    std::vector<unsigned char> payload;
  };
  std::vector<StreamingPlaintextPayloadChunk> streaming_plaintext_payload_chunks;
  bool metadata_cached = false;
  unsigned long long cached_device_bytes = 0;
  int cached_uses_streaming = 0;
  int cached_width = 0;
  std::vector<std::pair<int, int>> cached_rotation_key_requests;
  bool singleton = false;
  int singleton_diag_idx = 0;
  std::vector<Complex> singleton_values;
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

  LinearTransformState(cheddar::StripedMatrix raw_matrix,
                       std::vector<int> indices, int transform_level,
                       int diag_idx, std::vector<Complex> diag_values)
      : matrix(std::move(raw_matrix)),
        diag_indices(std::move(indices)),
        singleton(true),
        singleton_diag_idx(diag_idx),
        singleton_values(std::move(diag_values)),
        level(transform_level),
        bs(1),
        gs(1) {}

  LinearTransformState(std::vector<int> indices, int transform_level,
                       int width, int baby_step, int giant_step)
      : diag_indices(std::move(indices)),
        level(transform_level),
        bs(baby_step),
        gs(giant_step) {
    cached_width = std::max(1, width);
  }
};

struct SharedCacheBucket {
  int level = 0;
  std::vector<int> transform_ids;
  std::unique_ptr<HoistHandler> cache;
};

struct SharedCachePlan {
  std::vector<SharedCacheBucket> buckets;
};

struct SharedCacheEvalProfile {
  double plan_s = 0.0;
  double level_adjust_s = 0.0;
  double baby_step_s = 0.0;
  double giant_step_s = 0.0;
  double stream_build_map_s = 0.0;
  double stream_encode_hoist_s = 0.0;
  double stream_load_payload_s = 0.0;
  double stream_eval_s = 0.0;
  double stream_accumulate_s = 0.0;
  double push_s = 0.0;
  double trim_s = 0.0;
};

struct SchemeState {
  std::unique_ptr<Parameter> param;
  std::shared_ptr<BootContext> context;
  std::unique_ptr<UserInterface> interface;
  std::unique_ptr<UserInterface> bootstrap_interface;
  HeapAllocator<Pt> plaintexts;
  HeapAllocator<Ct> ciphertexts;
  HeapAllocator<PolynomialSpec> polynomials;
  HeapAllocator<LinearTransformState> transforms;
  std::map<std::vector<int>, SharedCachePlan> shared_cache_plans;
  std::map<int, int> prepared_rotation_key_levels;
  bool eval_mod_prepared = false;
  std::set<int> prepared_boot_slots;
};

std::unique_ptr<SchemeState> g_scheme;
SharedCacheEvalProfile g_shared_cache_eval_profile;

void AddDuration(double &target,
                 const std::chrono::steady_clock::time_point &started) {
  target +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
}

void AccumulateSharedCacheEvalProfile(const SharedCacheEvalProfile &profile) {
  g_shared_cache_eval_profile.plan_s += profile.plan_s;
  g_shared_cache_eval_profile.level_adjust_s += profile.level_adjust_s;
  g_shared_cache_eval_profile.baby_step_s += profile.baby_step_s;
  g_shared_cache_eval_profile.giant_step_s += profile.giant_step_s;
  g_shared_cache_eval_profile.stream_build_map_s += profile.stream_build_map_s;
  g_shared_cache_eval_profile.stream_encode_hoist_s +=
      profile.stream_encode_hoist_s;
  g_shared_cache_eval_profile.stream_load_payload_s +=
      profile.stream_load_payload_s;
  g_shared_cache_eval_profile.stream_eval_s += profile.stream_eval_s;
  g_shared_cache_eval_profile.stream_accumulate_s +=
      profile.stream_accumulate_s;
  g_shared_cache_eval_profile.push_s += profile.push_s;
  g_shared_cache_eval_profile.trim_s += profile.trim_s;
}

void SyncBootstrapInterfaceSecrets(SchemeState &state) {
  if (!state.interface) {
    return;
  }
  cheddar::HostVector<word> main_secret;
  cheddar::HostVector<word> sparse_secret;
  state.interface->ExportSecrets(main_secret, sparse_secret);
  state.bootstrap_interface = std::make_unique<UserInterface>(state.context);
  state.bootstrap_interface->LoadSecrets(main_secret, sparse_secret);
}

[[noreturn]] void AbortWithMessage(const char *message) {
  std::fprintf(stderr, "Cheddar backend error: %s\n", message);
  std::abort();
}

void RequireScheme() {
  if (!g_scheme) {
    AbortWithMessage("scheme is not initialized");
  }
}

void EnsureRotationKeyPrepared(int key, int level);
bool LinearTransformHasRotationKey(const LinearTransformState &state, int key);
const EvkMap &LinearTransformEvkMap(const LinearTransformState &state);
int SingletonLinearTransformRotationKey(const LinearTransformState &state);
bool PersistSharedCachePlans();
LinearTransformState MakeLinearTransformLevelView(
    const LinearTransformState &state, int eval_level);
void PrepareLinearTransformRotationKeysAtLevel(
    const LinearTransformState &state, int eval_level);

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
  SyncBootstrapInterfaceSecrets(*state);
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
    if (state.matrix.empty()) {
      AbortWithMessage(
          "linear transform matrix was released before plaintexts were loaded");
    }
    state.transform = std::make_unique<LinearTransform>(
        g_scheme->context, state.matrix, state.level,
        g_scheme->param->GetScale(state.level), state.bs, state.gs);
  }
  return *state.transform;
}

int LinearTransformWidth(const LinearTransformState &state) {
  if (!state.matrix.empty()) {
    return state.matrix.GetWidth();
  }
  return std::max(1, state.cached_width);
}

std::vector<int> LinearTransformDiagIndicesForLayout(
    const LinearTransformState &state) {
  if (state.matrix.empty()) {
    return state.diag_indices;
  }
  std::vector<int> indices;
  indices.reserve(state.matrix.size());
  for (const auto &[diag_idx, _] : state.matrix) {
    indices.push_back(diag_idx);
  }
  return indices;
}

int EstimateLinearTransformStride(const LinearTransformState &state) {
  const int width = LinearTransformWidth(state);
  int gcd_rot = 0;
  int max_rot = 0;
  int num_pt = 0;
  for (const int diag_idx : LinearTransformDiagIndicesForLayout(state)) {
    ++num_pt;
    int rot = diag_idx % width;
    if (rot < 0) {
      rot += width;
    }
    if (gcd_rot == 0 && rot == 0) {
      gcd_rot = rot;
    } else {
      gcd_rot = std::gcd(gcd_rot, rot);
    }
    max_rot = std::max(max_rot, rot);
  }
  if (num_pt <= 1 || gcd_rot <= 0) {
    return 1;
  }
  const int max_pt_dist = (state.bs * state.gs - 1) * gcd_rot;
  if (max_rot > max_pt_dist) {
    return 1;
  }
  return gcd_rot;
}

unsigned long long EstimateLinearTransformStateDeviceBytes(
    const LinearTransformState &state) {
  const int stride = EstimateLinearTransformStride(state);
  const int gs_stride = std::max(1, stride * state.bs);
  const int width = LinearTransformWidth(state);
  std::set<std::pair<int, int>> plaintext_slots;
  for (const int diag_idx : LinearTransformDiagIndicesForLayout(state)) {
    int rot = diag_idx % width;
    if (rot < 0) {
      rot += width;
    }
    const int bs_rot = rot % gs_stride;
    const int gs_rot = rot - bs_rot;
    plaintext_slots.emplace(gs_rot, bs_rot);
  }

  const auto np = g_scheme->param->LevelToNP(state.level, g_scheme->param->alpha_);
  const unsigned long long bytes_per_plaintext =
      static_cast<unsigned long long>(np.GetNumTotal()) *
      static_cast<unsigned long long>(g_scheme->param->degree_) *
      static_cast<unsigned long long>(sizeof(word));
  return static_cast<unsigned long long>(plaintext_slots.size()) *
         bytes_per_plaintext;
}

struct LinearTransformLayout {
  int stride = 1;
  int gs_stride = 1;
  std::set<int> baby_steps;
  std::vector<int> giant_steps;
};

LinearTransformLayout DescribeLinearTransformLayout(
    const LinearTransformState &state) {
  LinearTransformLayout layout;
  layout.stride = EstimateLinearTransformStride(state);
  layout.gs_stride = std::max(1, layout.stride * state.bs);
  const int width = LinearTransformWidth(state);
  std::set<int> giant_step_set;
  for (const int diag_idx : LinearTransformDiagIndicesForLayout(state)) {
    int rot = diag_idx % width;
    if (rot < 0) {
      rot += width;
    }
    const int bs_rot = rot % layout.gs_stride;
    const int gs_rot = rot - bs_rot;
    layout.baby_steps.insert(bs_rot);
    giant_step_set.insert(gs_rot);
  }
  if (layout.baby_steps.empty()) {
    layout.baby_steps.insert(0);
  }
  layout.giant_steps.assign(giant_step_set.begin(), giant_step_set.end());
  if (layout.giant_steps.empty()) {
    layout.giant_steps.push_back(0);
  }
  return layout;
}

void AddLinearTransformRequiredRotations(const LinearTransformState &state,
                                         cheddar::EvkRequest &req) {
  const LinearTransformLayout layout = DescribeLinearTransformLayout(state);
  for (const int bs_idx : layout.baby_steps) {
    if (bs_idx != 0) {
      req.AddRequest(bs_idx, state.level);
    }
  }
  for (const int gs_idx : layout.giant_steps) {
    if (gs_idx != 0) {
      req.AddRequest(gs_idx, state.level);
    }
  }
}

unsigned long long ReadULLFromEnv(const char *name,
                                  unsigned long long default_value) {
  const char *raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return default_value;
  }
  char *end = nullptr;
  const unsigned long long value = std::strtoull(raw, &end, 10);
  if (end == raw || value == 0) {
    return default_value;
  }
  return value;
}

unsigned long long ReadULLFromEnvOrZero(const char *name) {
  const char *raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return 0;
  }
  char *end = nullptr;
  const unsigned long long value = std::strtoull(raw, &end, 10);
  if (end == raw) {
    return 0;
  }
  return value;
}

bool EnvValueIsFalse(const char *value) {
  return value != nullptr &&
         (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 ||
          std::strcmp(value, "False") == 0 ||
          std::strcmp(value, "off") == 0 || std::strcmp(value, "OFF") == 0);
}

bool EnvValueIsTrue(const char *value) {
  return value != nullptr &&
         (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
          std::strcmp(value, "True") == 0 ||
          std::strcmp(value, "on") == 0 || std::strcmp(value, "ON") == 0 ||
          std::strcmp(value, "force") == 0);
}

bool StreamingPlaintextPayloadCacheEnabled() {
  return !EnvValueIsFalse(std::getenv("ORION_CHEDDAR_LT_STREAM_PAYLOAD_CACHE"));
}

bool StreamingPlaintextPayloadProfileEnabled() {
  return EnvValueIsTrue(std::getenv("ORION_CHEDDAR_LT_STREAM_PAYLOAD_PROFILE"));
}

bool StreamingPlaintextPayloadPipelineEnabled() {
  return EnvValueIsTrue(std::getenv("ORION_CHEDDAR_LT_STREAM_PAYLOAD_PIPELINE"));
}

unsigned long long LinearTransformFullLoadBudgetBytes() {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  const cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (err != cudaSuccess || free_bytes == 0 || total_bytes == 0) {
    return 0;
  }
  const unsigned long long free_u64 =
      static_cast<unsigned long long>(free_bytes);
  const unsigned long long total_u64 =
      static_cast<unsigned long long>(total_bytes);
  const unsigned long long default_reserve =
      std::max(16ULL * 1024ULL * 1024ULL * 1024ULL, total_u64 / 5ULL);
  const unsigned long long reserve = ReadULLFromEnv(
      "ORION_CHEDDAR_LT_FULL_LOAD_RESERVE_BYTES", default_reserve);
  const unsigned long long pct =
      std::min<unsigned long long>(
          95ULL, ReadULLFromEnv("ORION_CHEDDAR_LT_FULL_LOAD_BUDGET_PCT", 85ULL));
  const unsigned long long fraction_budget = free_u64 * pct / 100ULL;
  if (free_u64 <= reserve) {
    return std::max(256ULL * 1024ULL * 1024ULL, fraction_budget / 2ULL);
  }
  return std::min(free_u64 - reserve, fraction_budget);
}

bool ShouldStreamLinearTransform(const LinearTransformState &state) {
  if (state.metadata_cached && state.matrix.empty()) {
    return state.cached_uses_streaming != 0;
  }
  if (state.singleton) {
    return false;
  }
  const char *mode = std::getenv("ORION_CHEDDAR_LT_STREAMING");
  if (EnvValueIsFalse(mode)) {
    return false;
  }
  if (EnvValueIsTrue(mode)) {
    return true;
  }
  constexpr unsigned long long kDefaultStreamingThresholdBytes =
      8ULL * 1024ULL * 1024ULL * 1024ULL;
  const unsigned long long threshold = ReadULLFromEnv(
      "ORION_CHEDDAR_LT_STREAMING_THRESHOLD_BYTES",
      kDefaultStreamingThresholdBytes);
  const unsigned long long estimate = EstimateLinearTransformStateDeviceBytes(state);
  const unsigned long long budget = LinearTransformFullLoadBudgetBytes();
  if (budget > 0) {
    return estimate > budget;
  }
  return estimate >= threshold;
}

void CacheLinearTransformMetadata(LinearTransformState &state) {
  if (state.metadata_cached) {
    return;
  }
  state.cached_width = LinearTransformWidth(state);
  state.cached_rotation_key_requests.clear();
  if (state.singleton) {
    const int key = SingletonLinearTransformRotationKey(state);
    if (key != 0) {
      state.cached_rotation_key_requests.emplace_back(key, state.level);
    }
    state.cached_device_bytes =
        static_cast<unsigned long long>(state.singleton_values.size() *
                                        sizeof(Complex));
    state.cached_uses_streaming = 0;
    state.metadata_cached = true;
    return;
  }

  state.cached_device_bytes = EstimateLinearTransformStateDeviceBytes(state);
  state.cached_uses_streaming = ShouldStreamLinearTransform(state) ? 1 : 0;

  cheddar::EvkRequest req;
  AddLinearTransformRequiredRotations(state, req);
  state.cached_rotation_key_requests.reserve(req.size());
  for (const auto &[key, level] : req) {
    if (key != 0) {
      state.cached_rotation_key_requests.emplace_back(key, level);
    }
  }
  state.metadata_cached = true;
}

void ReleaseLinearTransformMatrixState(LinearTransformState &state) {
  if (state.singleton || state.matrix.empty()) {
    return;
  }
  CacheLinearTransformMetadata(state);
  state.transform.reset();
  state.matrix = cheddar::StripedMatrix();
}

unsigned long long LinearTransformPlaintextBytesPerSlot(
    const LinearTransformState &state) {
  const auto np = g_scheme->param->LevelToNP(state.level, g_scheme->param->alpha_);
  return static_cast<unsigned long long>(np.GetNumTotal()) *
         static_cast<unsigned long long>(g_scheme->param->degree_) *
         static_cast<unsigned long long>(sizeof(word));
}

std::map<int, unsigned long long> EstimatePlaintextBytesByGiantStep(
    const LinearTransformState &state, const LinearTransformLayout &layout) {
  std::map<int, std::set<int>> baby_steps_by_giant_step;
  const int width = LinearTransformWidth(state);
  for (const int diag_idx : LinearTransformDiagIndicesForLayout(state)) {
    int rot = diag_idx % width;
    if (rot < 0) {
      rot += width;
    }
    const int bs_rot = rot % layout.gs_stride;
    const int gs_rot = rot - bs_rot;
    baby_steps_by_giant_step[gs_rot].insert(bs_rot);
  }
  const unsigned long long bytes_per_plaintext =
      LinearTransformPlaintextBytesPerSlot(state);
  std::map<int, unsigned long long> result;
  for (const auto &[gs_idx, bs_set] : baby_steps_by_giant_step) {
    result[gs_idx] =
        static_cast<unsigned long long>(bs_set.size()) * bytes_per_plaintext;
  }
  return result;
}

unsigned long long CiphertextDeviceBytesForNP(const cheddar::NPInfo &np,
                                              bool has_rx = false) {
  const unsigned long long polys = has_rx ? 3ULL : 2ULL;
  return polys * static_cast<unsigned long long>(np.GetNumTotal()) *
         static_cast<unsigned long long>(g_scheme->param->degree_) *
         static_cast<unsigned long long>(sizeof(word));
}

unsigned long long LinearTransformCiphertextBytesAtLevel(
    const LinearTransformState &state) {
  const auto np = g_scheme->param->LevelToNP(state.level, g_scheme->param->alpha_);
  return CiphertextDeviceBytesForNP(np);
}

unsigned long long StreamingGiantStepMemoryBudgetBytes() {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  const cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (err != cudaSuccess || free_bytes == 0) {
    return 1ULL * 1024ULL * 1024ULL * 1024ULL;
  }
  const unsigned long long free_u64 =
      static_cast<unsigned long long>(free_bytes);
  const unsigned long long total_u64 =
      static_cast<unsigned long long>(total_bytes);
  const unsigned long long default_reserve =
      std::max(16ULL * 1024ULL * 1024ULL * 1024ULL, total_u64 / 4ULL);
  const unsigned long long reserve = ReadULLFromEnv(
      "ORION_CHEDDAR_LT_STREAM_RESERVE_BYTES", default_reserve);
  const unsigned long long pct =
      std::min<unsigned long long>(
          95ULL, ReadULLFromEnv("ORION_CHEDDAR_LT_STREAM_BUDGET_PCT", 45ULL));
  const unsigned long long fraction_budget =
      free_u64 * pct / 100ULL;
  if (free_u64 > reserve) {
    return std::min(free_u64 - reserve, fraction_budget);
  }
  return std::max(256ULL * 1024ULL * 1024ULL, fraction_budget / 2ULL);
}

std::vector<std::vector<int>> BuildStreamingGiantStepChunks(
    const LinearTransformState &state, const LinearTransformLayout &layout,
    const std::map<int, Ct> &bs_cache) {
  const unsigned long long fixed_chunk =
      ReadULLFromEnvOrZero("ORION_CHEDDAR_LT_STREAM_GS_CHUNK");
  const unsigned long long configured_max_chunk =
      ReadULLFromEnvOrZero("ORION_CHEDDAR_LT_STREAM_GS_MAX_CHUNK");
  const std::size_t max_chunk =
      static_cast<std::size_t>(std::max<unsigned long long>(
          1ULL,
          fixed_chunk > 0
              ? fixed_chunk
              : (configured_max_chunk > 0
                     ? configured_max_chunk
                     : static_cast<unsigned long long>(
                           layout.giant_steps.size()))));
  const auto bytes_by_gs = EstimatePlaintextBytesByGiantStep(state, layout);
  const unsigned long long ct_bytes =
      bs_cache.empty()
          ? LinearTransformCiphertextBytesAtLevel(state)
          : CiphertextDeviceBytesForNP(bs_cache.begin()->second.GetNP());
  const unsigned long long budget =
      fixed_chunk > 0 ? ~0ULL : StreamingGiantStepMemoryBudgetBytes();

  std::vector<std::vector<int>> chunks;
  std::vector<int> current;
  unsigned long long current_plaintext_bytes = 0;
  for (const int gs_idx : layout.giant_steps) {
    const auto it = bytes_by_gs.find(gs_idx);
    const unsigned long long gs_plaintext_bytes =
        it == bytes_by_gs.end() ? 0ULL : it->second;
    const unsigned long long candidate_plaintext_bytes =
        current_plaintext_bytes + gs_plaintext_bytes;
    const std::size_t candidate_count = current.size() + 1;
    const unsigned long long candidate_bytes =
        candidate_plaintext_bytes +
        (static_cast<unsigned long long>(candidate_count) + 4ULL) * ct_bytes;
    const bool over_count = candidate_count > max_chunk;
    const bool over_budget =
        fixed_chunk == 0 && !current.empty() && candidate_bytes > budget;
    if (over_count || over_budget) {
      chunks.push_back(std::move(current));
      current = {};
      current_plaintext_bytes = 0;
    }
    current.push_back(gs_idx);
    current_plaintext_bytes += gs_plaintext_bytes;
  }
  if (!current.empty()) {
    chunks.push_back(std::move(current));
  }
  if (chunks.empty()) {
    chunks.push_back(std::vector<int>{0});
  }
  if (EnvValueIsTrue(std::getenv("ORION_CHEDDAR_LT_STREAM_LOG"))) {
    std::fprintf(stderr,
                 "Cheddar streaming LT: %zu giant steps in %zu chunks "
                 "(max_chunk=%zu, budget=%llu, ct_bytes=%llu)\n",
                 layout.giant_steps.size(), chunks.size(), max_chunk, budget,
                 ct_bytes);
  }
  return chunks;
}

cheddar::PlainHoistMap BuildPlainHoistMapForGiantSteps(
    const LinearTransformState &state, const LinearTransformLayout &layout,
    const std::vector<int> &selected_giant_steps) {
  const std::set<int> selected(selected_giant_steps.begin(),
                               selected_giant_steps.end());
  const int height = state.matrix.GetHeight();
  const int width = state.matrix.GetWidth();
  cheddar::PlainHoistMap hoist_map;
  for (const auto &[diag_idx, diag] : state.matrix) {
    int rot = diag_idx % width;
    if (rot < 0) {
      rot += width;
    }
    const int bs_rot = rot % layout.gs_stride;
    const int gs_rot = rot - bs_rot;
    if (selected.find(gs_rot) == selected.end()) {
      continue;
    }
    auto &message = hoist_map[gs_rot][bs_rot];
    if (message.empty()) {
      message.assign(height, Complex(0.0, 0.0));
    }
    int offset = gs_rot % height;
    if (offset < 0) {
      offset += height;
    }
    for (int j = 0; j < height; ++j) {
      message[(j + offset) % height] = diag[j];
    }
  }
  return hoist_map;
}

void SynchronizeCudaAfterStreamingChunk() {
  const cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    AbortWithMessage("cuda synchronization failed after streaming LT chunk");
  }
}

int CiphertextDegree(const Ct &ciphertext) { return ciphertext.HasRx() ? 2 : 1; }

int CiphertextLevel(const Ct &ciphertext) {
  return g_scheme->param->NPToLevel(ciphertext.GetNP());
}

int PlaintextLevel(const Pt &plaintext) {
  return g_scheme->param->NPToLevel(plaintext.GetNP());
}

const Ct &CiphertextAtLevel(Ct &scratch, const Ct &ciphertext,
                            int target_level) {
  const int level = CiphertextLevel(ciphertext);
  if (level == target_level) {
    return ciphertext;
  }
  if (level < target_level) {
    AbortWithMessage("ciphertext level is below requested target level");
  }
  if (target_level < 0) {
    AbortWithMessage("level-down to Cheddar short base is not supported here");
  }
  g_scheme->context->LevelDown(scratch, ciphertext, target_level);
  scratch.SetScale(ciphertext.GetScale());
  return scratch;
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

int NormalizeRotationIndex(int index, int width) {
  const int modulus = std::max(1, width);
  int rot = index % modulus;
  if (rot < 0) {
    rot += modulus;
  }
  return rot;
}

int DefaultRotationWidth() {
  RequireScheme();
  return std::max(1, g_scheme->param->degree_ / 2);
}

int NormalizeCiphertextRotation(int ciphertext_id, int amount) {
  const Ct &ciphertext = RetrieveCiphertext(ciphertext_id);
  return NormalizeRotationIndex(amount, ciphertext.GetNumSlots());
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
  const std::size_t offset = buffer.size();
  buffer.resize(offset + sizeof(T));
  std::memcpy(buffer.data() + offset, &value, sizeof(T));
}

void AppendRawBytes(std::vector<unsigned char> &buffer,
                    const unsigned char *data, std::size_t size) {
  if (size == 0) {
    return;
  }
  const std::size_t offset = buffer.size();
  buffer.resize(offset + size);
  std::memcpy(buffer.data() + offset, data, size);
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

void CopySerializedWordsToDevice(cheddar::DeviceVector<word> &dst,
                                 const unsigned char *&cursor,
                                 const unsigned char *end,
                                 std::uint64_t word_count) {
  const std::size_t count = static_cast<std::size_t>(word_count);
  const std::size_t byte_count = count * sizeof(word);
  if (static_cast<std::size_t>(end - cursor) < byte_count) {
    AbortWithMessage("serialized word payload is truncated");
  }

  dst.resize(static_cast<int>(count));
  void *pinned = nullptr;
  const cudaError_t alloc_status = cudaHostAlloc(&pinned, byte_count, 0);
  if (alloc_status == cudaSuccess && pinned != nullptr) {
    std::memcpy(pinned, cursor, byte_count);
    cudaMemcpyAsync(dst.data(), pinned, byte_count, cudaMemcpyHostToDevice,
                    dst.stream());
    cudaStreamSynchronize(dst.stream());
    cudaFreeHost(pinned);
  } else {
    cheddar::HostVector<word> host(count);
    std::memcpy(host.data(), cursor, byte_count);
    cheddar::CopyHostToDevice(dst, host);
  }
  cursor += byte_count;
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
    const std::size_t byte_count = static_cast<std::size_t>(vec_size) * sizeof(word);
    if (static_cast<std::size_t>(end - cursor) < byte_count * 2) {
      AbortWithMessage("serialized evaluation key payload is truncated");
    }
    CopySerializedWordsToDevice(key.bx_.at(index), cursor, end, vec_size);
    CopySerializedWordsToDevice(key.ax_.at(index), cursor, end, vec_size);
  }
  return key;
}

int SerializedEvaluationKeyLevel(const unsigned char *data, std::size_t size) {
  const unsigned char *cursor = data;
  const unsigned char *end = data + size;
  const int num_main = ReadPod<std::int32_t>(cursor, end);
  const int num_ter = ReadPod<std::int32_t>(cursor, end);
  const int num_aux = ReadPod<std::int32_t>(cursor, end);
  return g_scheme->param->NPToLevel(
      cheddar::NPInfo(num_main, num_ter, num_aux));
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
  if (!g_scheme->bootstrap_interface) {
    g_scheme->bootstrap_interface =
        std::make_unique<UserInterface>(g_scheme->context);
  }
  g_scheme->bootstrap_interface->LoadSecrets(main_secret, sparse_secret);
  g_scheme->eval_mod_prepared = false;
  g_scheme->prepared_boot_slots.clear();
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

constexpr std::uint32_t kLinearTransformPlaintextsMagic = 0x4f484c54U;
constexpr std::uint32_t kLinearTransformPlaintextsVersion = 1U;

std::size_t SerializedPlaintextPayloadByteCount(const Pt &plaintext) {
  const cheddar::NPInfo np = plaintext.GetNP();
  const std::size_t word_count =
      static_cast<std::size_t>(np.GetNumTotal()) *
      static_cast<std::size_t>(g_scheme->param->degree_);
  return 4 * sizeof(std::int32_t) + sizeof(double) + sizeof(std::uint64_t) +
         word_count * sizeof(word);
}

void AppendPlaintextPayload(std::vector<unsigned char> &buffer,
                            const Pt &plaintext) {
  const cheddar::NPInfo np = plaintext.GetNP();
  cheddar::HostVector<word> host;
  cheddar::CopyDeviceToHost(host, plaintext.mx_);
  AppendPod(buffer, static_cast<std::int32_t>(np.num_main_));
  AppendPod(buffer, static_cast<std::int32_t>(np.num_ter_));
  AppendPod(buffer, static_cast<std::int32_t>(np.num_aux_));
  AppendPod(buffer, static_cast<std::int32_t>(plaintext.GetNumSlots()));
  AppendPod(buffer, plaintext.GetScale());
  AppendPod(buffer, static_cast<std::uint64_t>(host.size()));
  const unsigned char *raw =
      reinterpret_cast<const unsigned char *>(host.data());
  AppendRawBytes(buffer, raw, host.size() * sizeof(word));
}

Pt ReadPlaintextPayload(const unsigned char *&cursor,
                        const unsigned char *end) {
  const int num_main = ReadPod<std::int32_t>(cursor, end);
  const int num_ter = ReadPod<std::int32_t>(cursor, end);
  const int num_aux = ReadPod<std::int32_t>(cursor, end);
  const int num_slots = ReadPod<std::int32_t>(cursor, end);
  const double scale = ReadPod<double>(cursor, end);
  const std::uint64_t word_count = ReadPod<std::uint64_t>(cursor, end);
  const cheddar::NPInfo np(num_main, num_ter, num_aux);
  const std::uint64_t expected_word_count =
      static_cast<std::uint64_t>(np.GetNumTotal()) *
      static_cast<std::uint64_t>(g_scheme->param->degree_);
  if (word_count != expected_word_count) {
    AbortWithMessage("serialized plaintext payload has invalid size");
  }
  Pt plaintext(np);
  plaintext.SetNumSlots(num_slots);
  plaintext.SetScale(scale);
  CopySerializedWordsToDevice(plaintext.mx_, cursor, end, word_count);
  return plaintext;
}

bool HasStreamingPlaintextPayloadCache(const LinearTransformState &state);
void EnsureLinearTransformStreamingPlaintextPayloadCache(
    LinearTransformState &state);

struct SerializedPlaintextMapHeader {
  int level = 0;
  int bs = 0;
  int gs = 0;
  std::uint64_t record_count = 0;
  const unsigned char *records_begin = nullptr;
  const unsigned char *end = nullptr;
};

SerializedPlaintextMapHeader ReadSerializedPlaintextMapHeader(
    const unsigned char *&cursor, const unsigned char *end) {
  const std::uint32_t magic = ReadPod<std::uint32_t>(cursor, end);
  const std::uint32_t version = ReadPod<std::uint32_t>(cursor, end);
  if (magic != kLinearTransformPlaintextsMagic ||
      version != kLinearTransformPlaintextsVersion) {
    AbortWithMessage("unsupported linear transform plaintext payload");
  }
  SerializedPlaintextMapHeader header;
  header.level = ReadPod<std::int32_t>(cursor, end);
  header.bs = ReadPod<std::int32_t>(cursor, end);
  header.gs = ReadPod<std::int32_t>(cursor, end);
  header.record_count = ReadPod<std::uint64_t>(cursor, end);
  header.records_begin = cursor;
  header.end = end;
  return header;
}

void ValidatePlaintextMapHeader(const LinearTransformState &state,
                                const SerializedPlaintextMapHeader &header) {
  if (header.level != state.level || header.bs != state.bs ||
      header.gs != state.gs) {
    AbortWithMessage("linear transform plaintext payload does not match transform");
  }
}

void AppendPlaintextMapHeader(std::vector<unsigned char> &payload, int level,
                              int bs, int gs, std::uint64_t record_count) {
  AppendPod(payload, kLinearTransformPlaintextsMagic);
  AppendPod(payload, kLinearTransformPlaintextsVersion);
  AppendPod(payload, static_cast<std::int32_t>(level));
  AppendPod(payload, static_cast<std::int32_t>(bs));
  AppendPod(payload, static_cast<std::int32_t>(gs));
  AppendPod(payload, record_count);
}

void SkipPlaintextPayloadBytes(const unsigned char *&cursor,
                               const unsigned char *end) {
  const int num_main = ReadPod<std::int32_t>(cursor, end);
  const int num_ter = ReadPod<std::int32_t>(cursor, end);
  const int num_aux = ReadPod<std::int32_t>(cursor, end);
  (void)ReadPod<std::int32_t>(cursor, end);
  (void)ReadPod<double>(cursor, end);
  const std::uint64_t word_count = ReadPod<std::uint64_t>(cursor, end);
  const cheddar::NPInfo np(num_main, num_ter, num_aux);
  const std::uint64_t expected_word_count =
      static_cast<std::uint64_t>(np.GetNumTotal()) *
      static_cast<std::uint64_t>(g_scheme->param->degree_);
  if (word_count != expected_word_count) {
    AbortWithMessage("serialized plaintext payload has invalid size");
  }
  const std::uint64_t byte_count = word_count * sizeof(word);
  if (static_cast<std::uint64_t>(end - cursor) < byte_count) {
    AbortWithMessage("serialized word payload is truncated");
  }
  cursor += byte_count;
}

struct SerializedPlaintextRecordRef {
  int gs_idx = 0;
  int bs_idx = 0;
  const unsigned char *begin = nullptr;
  const unsigned char *end = nullptr;
};

SerializedPlaintextRecordRef ReadSerializedPlaintextRecordRef(
    const unsigned char *&cursor, const unsigned char *end) {
  SerializedPlaintextRecordRef record;
  record.begin = cursor;
  record.gs_idx = ReadPod<std::int32_t>(cursor, end);
  record.bs_idx = ReadPod<std::int32_t>(cursor, end);
  SkipPlaintextPayloadBytes(cursor, end);
  record.end = cursor;
  return record;
}

std::vector<unsigned char> SerializePlaintextMapBytes(
    int level, int bs, int gs,
    const LinearTransform::PlaintextMap &plaintext_map) {
  std::uint64_t record_count = 0;
  std::size_t payload_bytes = 2 * sizeof(std::uint32_t) +
                              3 * sizeof(std::int32_t) +
                              sizeof(std::uint64_t);
  for (const auto &[_, bs_map] : plaintext_map) {
    record_count += static_cast<std::uint64_t>(bs_map.size());
    for (const auto &[_, plaintext] : bs_map) {
      payload_bytes += 2 * sizeof(std::int32_t);
      payload_bytes += SerializedPlaintextPayloadByteCount(plaintext);
    }
  }

  std::vector<unsigned char> payload;
  payload.reserve(payload_bytes);
  AppendPlaintextMapHeader(payload, level, bs, gs, record_count);
  for (const auto &[gs_idx, bs_map] : plaintext_map) {
    for (const auto &[bs_idx, plaintext] : bs_map) {
      AppendPod(payload, static_cast<std::int32_t>(gs_idx));
      AppendPod(payload, static_cast<std::int32_t>(bs_idx));
      AppendPlaintextPayload(payload, plaintext);
    }
  }
  return payload;
}

LinearTransform::PlaintextMap DeserializePlaintextMapBytes(
    const LinearTransformState &state, const unsigned char *data,
    std::size_t size) {
  const unsigned char *cursor = data;
  const unsigned char *end = data + size;
  const SerializedPlaintextMapHeader header =
      ReadSerializedPlaintextMapHeader(cursor, end);
  ValidatePlaintextMapHeader(state, header);

  LinearTransform::PlaintextMap plaintext_map;
  for (std::uint64_t record = 0; record < header.record_count; ++record) {
    const int gs_idx = ReadPod<std::int32_t>(cursor, end);
    const int bs_idx = ReadPod<std::int32_t>(cursor, end);
    plaintext_map[gs_idx].emplace(bs_idx, ReadPlaintextPayload(cursor, end));
  }
  if (cursor != end) {
    AbortWithMessage("linear transform plaintext payload has trailing bytes");
  }
  return plaintext_map;
}

std::vector<unsigned char> SerializeStreamingPlaintextPayloadChunksBytes(
    const LinearTransformState &state) {
  std::uint64_t record_count = 0;
  std::size_t payload_bytes = 2 * sizeof(std::uint32_t) +
                              3 * sizeof(std::int32_t) +
                              sizeof(std::uint64_t);
  std::vector<std::pair<const unsigned char *, const unsigned char *>>
      record_ranges;
  record_ranges.reserve(state.streaming_plaintext_payload_chunks.size());
  for (const auto &chunk : state.streaming_plaintext_payload_chunks) {
    const unsigned char *cursor = chunk.payload.data();
    const unsigned char *end = chunk.payload.data() + chunk.payload.size();
    const SerializedPlaintextMapHeader header =
        ReadSerializedPlaintextMapHeader(cursor, end);
    ValidatePlaintextMapHeader(state, header);
    record_count += header.record_count;
    payload_bytes += static_cast<std::size_t>(end - header.records_begin);
    record_ranges.emplace_back(header.records_begin, end);
    cursor = header.records_begin;
    for (std::uint64_t record = 0; record < header.record_count; ++record) {
      (void)ReadSerializedPlaintextRecordRef(cursor, end);
    }
    if (cursor != end) {
      AbortWithMessage("linear transform plaintext payload has trailing bytes");
    }
  }

  std::vector<unsigned char> payload;
  payload.reserve(payload_bytes);
  AppendPlaintextMapHeader(payload, state.level, state.bs, state.gs,
                           record_count);
  for (const auto &[begin, end] : record_ranges) {
    AppendRawBytes(payload, begin, static_cast<std::size_t>(end - begin));
  }
  return payload;
}

std::vector<unsigned char> SerializeLinearTransformPlaintextsBytes(
    LinearTransformState &state) {
  if (ShouldStreamLinearTransform(state)) {
    EnsureLinearTransformStreamingPlaintextPayloadCache(state);
    if (HasStreamingPlaintextPayloadCache(state)) {
      return SerializeStreamingPlaintextPayloadChunksBytes(state);
    }
  }
  const LinearTransform &transform = EnsureTransformLoaded(state);
  return SerializePlaintextMapBytes(state.level, state.bs, state.gs,
                                    transform.GetPlaintextMap());
}

void LoadLinearTransformStreamingPlaintextsBytes(LinearTransformState &state,
                                                 const unsigned char *data,
                                                 std::size_t size) {
  const LinearTransformLayout layout = DescribeLinearTransformLayout(state);
  const std::vector<std::vector<int>> chunks =
      BuildStreamingGiantStepChunks(state, layout, {});
  std::map<int, std::size_t> chunk_index_by_gs;
  for (std::size_t chunk_index = 0; chunk_index < chunks.size(); ++chunk_index) {
    for (const int gs_idx : chunks[chunk_index]) {
      chunk_index_by_gs[gs_idx] = chunk_index;
    }
  }
  std::vector<std::vector<SerializedPlaintextRecordRef>> records_by_chunk(
      chunks.size());

  const unsigned char *cursor = data;
  const unsigned char *end = data + size;
  const SerializedPlaintextMapHeader header =
      ReadSerializedPlaintextMapHeader(cursor, end);
  ValidatePlaintextMapHeader(state, header);
  for (std::uint64_t record = 0; record < header.record_count; ++record) {
    SerializedPlaintextRecordRef record_ref =
        ReadSerializedPlaintextRecordRef(cursor, end);
    const auto chunk_it = chunk_index_by_gs.find(record_ref.gs_idx);
    if (chunk_it == chunk_index_by_gs.end()) {
      AbortWithMessage("linear transform plaintext payload has unexpected giant step");
    }
    records_by_chunk[chunk_it->second].push_back(record_ref);
  }
  if (cursor != end) {
    AbortWithMessage("linear transform plaintext payload has trailing bytes");
  }

  state.streaming_plaintext_payload_chunks.clear();
  state.streaming_plaintext_payload_chunks.reserve(chunks.size());
  for (std::size_t chunk_index = 0; chunk_index < chunks.size(); ++chunk_index) {
    const auto &records = records_by_chunk[chunk_index];
    if (records.empty()) {
      continue;
    }
    LinearTransformState::StreamingPlaintextPayloadChunk cached_chunk;
    cached_chunk.giant_steps = chunks[chunk_index];
    std::size_t chunk_payload_bytes = 2 * sizeof(std::uint32_t) +
                                      3 * sizeof(std::int32_t) +
                                      sizeof(std::uint64_t);
    for (const SerializedPlaintextRecordRef &record : records) {
      chunk_payload_bytes += static_cast<std::size_t>(record.end - record.begin);
    }
    cached_chunk.payload.reserve(chunk_payload_bytes);
    AppendPlaintextMapHeader(cached_chunk.payload, state.level, state.bs,
                             state.gs,
                             static_cast<std::uint64_t>(records.size()));
    for (const SerializedPlaintextRecordRef &record : records) {
      AppendRawBytes(cached_chunk.payload, record.begin,
                     static_cast<std::size_t>(record.end - record.begin));
    }
    state.streaming_plaintext_payload_chunks.push_back(std::move(cached_chunk));
  }
  CacheLinearTransformMetadata(state);
  state.transform.reset();
  state.matrix = cheddar::StripedMatrix();
  if (!PersistSharedCachePlans()) {
    g_scheme->shared_cache_plans.clear();
  }
}

void LoadLinearTransformPlaintextsBytes(LinearTransformState &state,
                                        const unsigned char *data,
                                        std::size_t size) {
  if (ShouldStreamLinearTransform(state)) {
    LoadLinearTransformStreamingPlaintextsBytes(state, data, size);
    return;
  }
  LinearTransform::PlaintextMap plaintext_map =
      DeserializePlaintextMapBytes(state, data, size);

  state.transform.reset();
  state.transform = std::make_unique<LinearTransform>(
      g_scheme->context, std::move(plaintext_map), state.level,
      g_scheme->param->GetScale(state.level), state.bs, state.gs, 0, 0,
      EstimateLinearTransformStride(state));
  if (!PersistSharedCachePlans()) {
    g_scheme->shared_cache_plans.clear();
  }
}

bool HasStreamingPlaintextPayloadCache(const LinearTransformState &state) {
  return !state.streaming_plaintext_payload_chunks.empty();
}

void EnsureLinearTransformStreamingPlaintextPayloadCache(
    LinearTransformState &state) {
  if (!StreamingPlaintextPayloadCacheEnabled() || state.singleton ||
      HasStreamingPlaintextPayloadCache(state) || !ShouldStreamLinearTransform(state)) {
    return;
  }
  if (state.matrix.empty()) {
    AbortWithMessage("streaming plaintext payload cache requires raw matrix");
  }

  const LinearTransformLayout layout = DescribeLinearTransformLayout(state);
  const std::vector<std::vector<int>> chunks =
      BuildStreamingGiantStepChunks(state, layout, {});
  const bool profile_enabled = StreamingPlaintextPayloadProfileEnabled();
  const bool pipeline_enabled =
      StreamingPlaintextPayloadPipelineEnabled() && chunks.size() > 1;
  const auto total_started = std::chrono::steady_clock::now();
  double build_map_s = 0.0;
  double hoist_encode_s = 0.0;
  double serialize_s = 0.0;
  std::size_t nonempty_chunks = 0;
  std::size_t payload_bytes = 0;
  std::size_t max_payload_bytes = 0;

  struct ChunkBuildResult {
    cheddar::PlainHoistMap map;
    double seconds = 0.0;
  };

  auto build_chunk_map = [&](std::size_t chunk_index) -> ChunkBuildResult {
    const auto build_started = std::chrono::steady_clock::now();
    ChunkBuildResult result;
    result.map = BuildPlainHoistMapForGiantSteps(
        state, layout, chunks.at(chunk_index));
    result.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      build_started)
            .count();
    return result;
  };

  state.streaming_plaintext_payload_chunks.clear();
  state.streaming_plaintext_payload_chunks.reserve(chunks.size());
  std::future<ChunkBuildResult> next_build;
  if (pipeline_enabled) {
    next_build =
        std::async(std::launch::async, build_chunk_map, std::size_t{0});
  }
  for (std::size_t chunk_index = 0; chunk_index < chunks.size();
       ++chunk_index) {
    ChunkBuildResult chunk_build;
    if (pipeline_enabled) {
      chunk_build = next_build.get();
      if (chunk_index + 1 < chunks.size()) {
        next_build = std::async(std::launch::async, build_chunk_map,
                                chunk_index + 1);
      }
    } else {
      chunk_build = build_chunk_map(chunk_index);
    }
    build_map_s += chunk_build.seconds;
    cheddar::PlainHoistMap &chunk_map = chunk_build.map;
    if (chunk_map.empty()) {
      continue;
    }
    ++nonempty_chunks;
    const auto hoist_started = std::chrono::steady_clock::now();
    HoistHandler chunk_hoist(g_scheme->context, chunk_map, state.level,
                             g_scheme->param->GetScale(state.level), true);
    hoist_encode_s +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      hoist_started)
            .count();
    LinearTransformState::StreamingPlaintextPayloadChunk cached_chunk;
    cached_chunk.giant_steps = chunks.at(chunk_index);
    const auto serialize_started = std::chrono::steady_clock::now();
    cached_chunk.payload = SerializePlaintextMapBytes(
        state.level, state.bs, state.gs, chunk_hoist.GetPlaintextMap());
    serialize_s +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      serialize_started)
            .count();
    payload_bytes += cached_chunk.payload.size();
    max_payload_bytes = std::max(max_payload_bytes, cached_chunk.payload.size());
    state.streaming_plaintext_payload_chunks.push_back(std::move(cached_chunk));
  }
  if (profile_enabled) {
    const double total_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      total_started)
            .count();
    std::fprintf(
        stderr,
        "Cheddar streaming payload compile profile: chunks=%zu nonempty=%zu "
        "pipeline=%d build_map_s=%.6f hoist_encode_s=%.6f "
        "serialize_s=%.6f total_s=%.6f payload_bytes=%zu "
        "max_payload_bytes=%zu diag_count=%zu bs=%d gs=%d level=%d\n",
        chunks.size(), nonempty_chunks, pipeline_enabled ? 1 : 0, build_map_s,
        hoist_encode_s, serialize_s, total_s, payload_bytes, max_payload_bytes,
        state.matrix.size(), state.bs, state.gs, state.level);
    std::fflush(stderr);
  }
  CacheLinearTransformMetadata(state);
  state.transform.reset();
  state.matrix = cheddar::StripedMatrix();
}

LinearTransform::PlaintextMap LoadStreamingPlaintextPayloadChunk(
    const LinearTransformState &state,
    const LinearTransformState::StreamingPlaintextPayloadChunk &chunk) {
  return DeserializePlaintextMapBytes(state, chunk.payload.data(),
                                      chunk.payload.size());
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
  const Ct &lhs = RetrieveCiphertext(lhs_id);
  const Ct &rhs = RetrieveCiphertext(rhs_id);
  const int target_level = std::min(CiphertextLevel(lhs), CiphertextLevel(rhs));
  Ct lhs_leveled;
  Ct rhs_leveled;
  fn(result, CiphertextAtLevel(lhs_leveled, lhs, target_level),
     CiphertextAtLevel(rhs_leveled, rhs, target_level));
  RetrieveCiphertext(lhs_id) = std::move(result);
  return lhs_id;
}

template <typename Fn>
int ApplyCiphertextBinaryNew(int lhs_id, int rhs_id, Fn &&fn) {
  Ct result;
  const Ct &lhs = RetrieveCiphertext(lhs_id);
  const Ct &rhs = RetrieveCiphertext(rhs_id);
  const int target_level = std::min(CiphertextLevel(lhs), CiphertextLevel(rhs));
  Ct lhs_leveled;
  Ct rhs_leveled;
  fn(result, CiphertextAtLevel(lhs_leveled, lhs, target_level),
     CiphertextAtLevel(rhs_leveled, rhs, target_level));
  return PushCiphertext(std::move(result));
}

template <typename Fn>
int ApplyCiphertextPlainBinaryInPlace(int ciphertext_id, int plaintext_id,
                                      Fn &&fn) {
  Ct result;
  const Ct &ciphertext = RetrieveCiphertext(ciphertext_id);
  const Pt &plaintext = RetrievePlaintext(plaintext_id);
  const int ciphertext_level = CiphertextLevel(ciphertext);
  const int plaintext_level = PlaintextLevel(plaintext);
  if (ciphertext_level < plaintext_level) {
    AbortWithMessage("plaintext level exceeds ciphertext level");
  }
  Ct leveled;
  fn(result, CiphertextAtLevel(leveled, ciphertext, plaintext_level), plaintext);
  RetrieveCiphertext(ciphertext_id) = std::move(result);
  return ciphertext_id;
}

template <typename Fn>
int ApplyCiphertextPlainBinaryNew(int ciphertext_id, int plaintext_id,
                                  Fn &&fn) {
  Ct result;
  const Ct &ciphertext = RetrieveCiphertext(ciphertext_id);
  const Pt &plaintext = RetrievePlaintext(plaintext_id);
  const int ciphertext_level = CiphertextLevel(ciphertext);
  const int plaintext_level = PlaintextLevel(plaintext);
  if (ciphertext_level < plaintext_level) {
    AbortWithMessage("plaintext level exceeds ciphertext level");
  }
  Ct leveled;
  fn(result, CiphertextAtLevel(leveled, ciphertext, plaintext_level), plaintext);
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
    matrix[NormalizeRotationIndex(diag_idxs[diag_index], slots)] =
        std::move(values);
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
    matrix[NormalizeRotationIndex(diag_idxs[diag_index], slots)] =
        std::move(values);
  }
  return matrix;
}

std::pair<int, int> ChooseLinearTransformSplit(
    const std::vector<int> &diag_indices, int width, float bsgs_ratio) {
  int stride = 0;
  int max_rot = 0;
  for (const int diag_idx : diag_indices) {
    const int rot = NormalizeRotationIndex(diag_idx, width);
    if (rot != 0) {
      stride = stride == 0 ? rot : std::gcd(stride, rot);
    }
    max_rot = std::max(max_rot, rot);
  }
  if (stride == 0) {
    stride = 1;
  }

  const int coverage = std::max(2, max_rot / stride + 1);
  const double ratio =
      std::isfinite(static_cast<double>(bsgs_ratio)) && bsgs_ratio > 0.0f
          ? static_cast<double>(bsgs_ratio)
          : 1.0;
  constexpr int kMaxBabyStepsForFusedPath = 128;
  int bs = static_cast<int>(std::ceil(std::sqrt(coverage * ratio)));
  bs = std::max(2, std::min(kMaxBabyStepsForFusedPath, bs));
  int gs = (coverage + bs - 1) / bs;
  gs = std::max(1, gs);
  while (bs * gs < coverage) {
    ++gs;
  }
  return {bs, gs};
}

std::pair<int, int> ChooseLinearTransformSplit(
    const cheddar::StripedMatrix &matrix, float bsgs_ratio) {
  std::vector<int> diag_indices;
  diag_indices.reserve(matrix.size());
  for (const auto &[diag_idx, _] : matrix) {
    diag_indices.push_back(diag_idx);
  }
  return ChooseLinearTransformSplit(diag_indices, matrix.GetWidth(), bsgs_ratio);
}

int AddLinearTransformFromMatrix(const cheddar::StripedMatrix &matrix,
                                 int level, float bsgs_ratio) {
  if (level < 0 || level > g_scheme->param->default_encryption_level_) {
    throw std::runtime_error(
        "linear transform level is outside supported preset range");
  }
  if (matrix.size() == 1) {
    const auto &entry = *matrix.begin();
    std::vector<int> diag_indices = {entry.first};
    return g_scheme->transforms.Add(LinearTransformState(
        matrix, std::move(diag_indices), level, entry.first, entry.second));
  }
  const auto [bs, gs] = ChooseLinearTransformSplit(matrix, bsgs_ratio);
  std::vector<int> diag_indices;
  diag_indices.reserve(matrix.size());
  for (const auto &[diag_idx, _] : matrix) {
    diag_indices.push_back(diag_idx);
  }
  LinearTransformState state(std::unique_ptr<LinearTransform>(), matrix,
                             std::move(diag_indices), level, bs, gs);
  EnsureLinearTransformStreamingPlaintextPayloadCache(state);
  return g_scheme->transforms.Add(std::move(state));
}

int AddLinearTransformFromDescriptor(const int *diag_idxs, int diag_count,
                                     int width, int level,
                                     float bsgs_ratio) {
  if (level < 0 || level > g_scheme->param->default_encryption_level_) {
    throw std::runtime_error(
        "linear transform level is outside supported preset range");
  }
  std::vector<int> diag_indices;
  diag_indices.reserve(std::max(0, diag_count));
  const int normalized_width = std::max(1, width);
  for (int i = 0; i < diag_count; ++i) {
    diag_indices.push_back(NormalizeRotationIndex(diag_idxs[i], normalized_width));
  }
  const auto [bs, gs] =
      ChooseLinearTransformSplit(diag_indices, normalized_width, bsgs_ratio);
  LinearTransformState state(std::move(diag_indices), level, normalized_width,
                             bs, gs);
  CacheLinearTransformMetadata(state);
  return g_scheme->transforms.Add(std::move(state));
}

int DefaultLinearTransformDescriptorWidth() {
  return std::max(1, g_scheme->param->degree_ / 2);
}

int SingletonLinearTransformRotationKey(const LinearTransformState &state) {
  if (!state.singleton) {
    return 0;
  }
  const int width = std::max(1, state.matrix.GetWidth());
  return NormalizeRotationIndex(state.singleton_diag_idx, width);
}

int EvaluateSingletonLinearTransform(const LinearTransformState &state,
                                     int ciphertext_id) {
  const Ct &input = RetrieveCiphertext(ciphertext_id);
  const int input_level = CiphertextLevel(input);
  if (input_level < state.level) {
    AbortWithMessage("input ciphertext level is below the linear transform level");
  }
  Ct leveled_input;
  const Ct &eval_input = CiphertextAtLevel(leveled_input, input, state.level);

  const int rotation = SingletonLinearTransformRotationKey(state);
  Ct rotated;
  const Ct *mul_input = &eval_input;
  if (rotation != 0) {
    EnsureRotationKeyPrepared(rotation, state.level);
    g_scheme->context->HRot(rotated, eval_input,
                            g_scheme->interface->GetRotationKey(rotation),
                            rotation);
    mul_input = &rotated;
  }

  Pt plaintext =
      EncodeMessage(state.singleton_values, state.level,
                    g_scheme->param->GetScale(state.level));
  Ct multiplied;
  g_scheme->context->Mult(multiplied, *mul_input, plaintext);
  Ct output;
  g_scheme->context->Rescale(output, multiplied);
  return PushCiphertext(std::move(output));
}

bool HasSingletonLinearTransform(const std::vector<int> &ordered_ids) {
  for (int transform_id : ordered_ids) {
    if (RetrieveTransform(transform_id).singleton) {
      return true;
    }
  }
  return false;
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

void PrepareLinearTransformRotationKeys(const LinearTransformState &state) {
  cheddar::EvkRequest req;
  AddLinearTransformRequiredRotations(state, req);
  for (const auto &[rot_idx, _] : req) {
    if (rot_idx != 0 && !LinearTransformHasRotationKey(state, rot_idx)) {
      EnsureRotationKeyPrepared(rot_idx, state.level);
    }
  }
}

void PrepareLinearTransformRotationKeysAtLevel(
    const LinearTransformState &state, int eval_level) {
  const LinearTransformLayout layout = DescribeLinearTransformLayout(state);
  for (const int bs_idx : layout.baby_steps) {
    if (bs_idx != 0 && !LinearTransformHasRotationKey(state, bs_idx)) {
      EnsureRotationKeyPrepared(bs_idx, eval_level);
    }
  }
  for (const int gs_idx : layout.giant_steps) {
    if (gs_idx != 0 && !LinearTransformHasRotationKey(state, gs_idx)) {
      EnsureRotationKeyPrepared(gs_idx, eval_level);
    }
  }
}

void EvaluateLinearTransformGiantStepStreaming(
    const LinearTransformState &state, Ct &output,
    const std::map<int, Ct> &bs_cache, const EvkMap &evk_map,
    SharedCacheEvalProfile *profile = nullptr) {
  const LinearTransformLayout layout = DescribeLinearTransformLayout(state);
  const bool use_cached_payloads = HasStreamingPlaintextPayloadCache(state);
  const std::vector<std::vector<int>> chunks =
      use_cached_payloads ? std::vector<std::vector<int>>()
                          : BuildStreamingGiantStepChunks(state, layout, bs_cache);
  if (use_cached_payloads &&
      EnvValueIsTrue(std::getenv("ORION_CHEDDAR_LT_STREAM_LOG"))) {
    std::fprintf(stderr,
                 "Cheddar cached streaming LT: %zu giant steps in %zu chunks\n",
                 layout.giant_steps.size(),
                 state.streaming_plaintext_payload_chunks.size());
  }
  bool initialized = false;
  Ct accumulated_output;
  const std::size_t chunk_count =
      use_cached_payloads ? state.streaming_plaintext_payload_chunks.size()
                          : chunks.size();
  for (std::size_t chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
    Ct partial_output;
    if (use_cached_payloads) {
      const auto load_started = std::chrono::steady_clock::now();
      LinearTransform::PlaintextMap plaintext_map = LoadStreamingPlaintextPayloadChunk(
          state, state.streaming_plaintext_payload_chunks.at(chunk_index));
      HoistHandler chunk_hoist(g_scheme->context, std::move(plaintext_map),
                               state.level,
                               g_scheme->param->GetScale(state.level));
      if (profile != nullptr) {
        AddDuration(profile->stream_load_payload_s, load_started);
      }
      const auto eval_started = std::chrono::steady_clock::now();
      chunk_hoist.EvaluateGiantStep(g_scheme->context, partial_output, bs_cache,
                                    evk_map, kUseMinKSLinearTransforms);
      SynchronizeCudaAfterStreamingChunk();
      if (profile != nullptr) {
        AddDuration(profile->stream_eval_s, eval_started);
      }
    } else {
      const std::vector<int> &chunk = chunks.at(chunk_index);
      const auto build_started = std::chrono::steady_clock::now();
      cheddar::PlainHoistMap chunk_map =
          BuildPlainHoistMapForGiantSteps(state, layout, chunk);
      if (profile != nullptr) {
        AddDuration(profile->stream_build_map_s, build_started);
      }
      if (chunk_map.empty()) {
        continue;
      }
      const auto encode_started = std::chrono::steady_clock::now();
      HoistHandler chunk_hoist(g_scheme->context, chunk_map, state.level,
                               g_scheme->param->GetScale(state.level), true);
      if (profile != nullptr) {
        AddDuration(profile->stream_encode_hoist_s, encode_started);
      }
      const auto eval_started = std::chrono::steady_clock::now();
      chunk_hoist.EvaluateGiantStep(g_scheme->context, partial_output, bs_cache,
                                    evk_map, kUseMinKSLinearTransforms);
      SynchronizeCudaAfterStreamingChunk();
      if (profile != nullptr) {
        AddDuration(profile->stream_eval_s, eval_started);
      }
    }
    const auto accumulate_started = std::chrono::steady_clock::now();
    if (!initialized) {
      accumulated_output = std::move(partial_output);
      initialized = true;
    } else {
      Ct sum;
      g_scheme->context->Add(sum, accumulated_output, partial_output);
      accumulated_output = std::move(sum);
    }
    if (profile != nullptr) {
      AddDuration(profile->stream_accumulate_s, accumulate_started);
    }
  }
  if (!initialized) {
    AbortWithMessage("streaming linear transform produced no giant-step chunks");
  }
  output = std::move(accumulated_output);
}

int EvaluateLinearTransformStreaming(LinearTransformState &state,
                                     int ciphertext_id) {
  const Ct &input = RetrieveCiphertext(ciphertext_id);
  const int input_level = CiphertextLevel(input);
  if (input_level < state.level) {
    AbortWithMessage("input ciphertext level is below the linear transform level");
  }
  Ct leveled_input;
  const Ct &eval_input = CiphertextAtLevel(leveled_input, input, state.level);
  const LinearTransformLayout layout = DescribeLinearTransformLayout(state);
  std::unique_ptr<HoistHandler> cache =
      BuildSharedCacheHoist(state.level, layout.baby_steps);
  std::map<int, Ct> bs_cache;
  cache->EvaluateBabyStep(g_scheme->context, bs_cache, eval_input,
                          LinearTransformEvkMap(state),
                          kUseMinKSLinearTransforms);
  Ct output;
  EvaluateLinearTransformGiantStepStreaming(
      state, output, bs_cache, LinearTransformEvkMap(state));
  return PushCiphertext(std::move(output));
}

SharedCachePlan BuildSharedCachePlan(const std::vector<int> &ordered_ids,
                                     int max_eval_level = -1) {
  SharedCachePlan plan;
  std::map<int, std::size_t> bucket_index_by_level;
  std::map<int, std::set<int>> bucket_bs_union;

  for (int transform_id : ordered_ids) {
    LinearTransformState &state = RetrieveTransform(transform_id);
    const int bucket_level =
        max_eval_level >= 0 ? std::min(state.level, max_eval_level) : state.level;
    std::unique_ptr<LinearTransformState> lowered_state;
    LinearTransformState *eval_state = &state;
    if (bucket_level != state.level) {
      lowered_state =
          std::make_unique<LinearTransformState>(
              MakeLinearTransformLevelView(state, bucket_level));
      eval_state = lowered_state.get();
    }
    auto [it, inserted] =
        bucket_index_by_level.emplace(bucket_level, plan.buckets.size());
    if (inserted) {
      SharedCacheBucket bucket;
      bucket.level = bucket_level;
      plan.buckets.push_back(std::move(bucket));
    }
    SharedCacheBucket &bucket = plan.buckets[it->second];
    bucket.transform_ids.push_back(transform_id);
    std::set<int> &union_indices = bucket_bs_union[bucket_level];
    union_indices.insert(0);
    if (ShouldStreamLinearTransform(*eval_state)) {
      const LinearTransformLayout layout = DescribeLinearTransformLayout(*eval_state);
      for (int bs_idx : layout.baby_steps) {
        union_indices.insert(bs_idx);
      }
      PrepareLinearTransformRotationKeysAtLevel(*eval_state, bucket_level);
    } else {
      LinearTransform &transform = EnsureTransformLoaded(*eval_state);
      for (int bs_idx : transform.GetBabyStepIndices()) {
        union_indices.insert(bs_idx);
      }
      cheddar::EvkRequest req;
      transform.AddRequiredRotations(req, kUseMinKSLinearTransforms);
      for (const auto &[rot_idx, _] : req) {
        if (rot_idx != 0 && !LinearTransformHasRotationKey(*eval_state, rot_idx)) {
          EnsureRotationKeyPrepared(rot_idx, bucket_level);
        }
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

bool PersistSharedCachePlans() {
  return EnvValueIsTrue(std::getenv("ORION_CHEDDAR_SHARED_CACHE_PLAN_PERSIST"));
}

LinearTransformState MakeLinearTransformLevelView(
    const LinearTransformState &state, int eval_level) {
  if (state.matrix.empty()) {
    AbortWithMessage(
        "cannot lower shared-cache transform level after matrix release");
  }
  if (state.singleton) {
    return LinearTransformState(state.matrix, state.diag_indices, eval_level,
                                state.singleton_diag_idx,
                                state.singleton_values);
  }
  return LinearTransformState(std::unique_ptr<LinearTransform>(), state.matrix,
                              state.diag_indices, eval_level, state.bs,
                              state.gs);
}

std::set<int> SharedCacheBabyStepsForLevel(
    const SharedCacheBucket &bucket, int eval_level) {
  std::set<int> baby_steps;
  baby_steps.insert(0);
  for (int transform_id : bucket.transform_ids) {
    LinearTransformState &state = RetrieveTransform(transform_id);
    std::unique_ptr<LinearTransformState> lowered_state;
    LinearTransformState *eval_state = &state;
    if (state.level != eval_level) {
      lowered_state =
          std::make_unique<LinearTransformState>(
              MakeLinearTransformLevelView(state, eval_level));
      eval_state = lowered_state.get();
    }
    const LinearTransformLayout layout =
        DescribeLinearTransformLayout(*eval_state);
    baby_steps.insert(layout.baby_steps.begin(), layout.baby_steps.end());
  }
  return baby_steps;
}

bool TrimDeviceMemoryAfterEval() {
  return EnvValueIsTrue(std::getenv("ORION_CHEDDAR_TRIM_AFTER_EVAL"));
}

unsigned long long DeviceMemoryTrimTargetBytes() {
  return ReadULLFromEnvOrZero("ORION_CHEDDAR_TRIM_TARGET_BYTES");
}

unsigned long long DeviceMemoryTrimMinFreeBytes(
    unsigned long long total_bytes) {
  const unsigned long long explicit_min =
      ReadULLFromEnvOrZero("ORION_CHEDDAR_TRIM_MIN_FREE_BYTES");
  if (explicit_min > 0) {
    return explicit_min;
  }
  const unsigned long long pct = std::min<unsigned long long>(
      95ULL, ReadULLFromEnv("ORION_CHEDDAR_TRIM_MIN_FREE_PCT", 20ULL));
  const unsigned long long pct_bytes = total_bytes * pct / 100ULL;
  return std::max(16ULL * 1024ULL * 1024ULL * 1024ULL, pct_bytes);
}

unsigned long long DeviceMemoryTrimDefaultTargetBytes(
    unsigned long long total_bytes) {
  const unsigned long long explicit_target = DeviceMemoryTrimTargetBytes();
  if (explicit_target > 0) {
    return explicit_target;
  }
  const unsigned long long pct = std::min<unsigned long long>(
      95ULL, ReadULLFromEnv("ORION_CHEDDAR_TRIM_TARGET_PCT", 12ULL));
  const unsigned long long pct_bytes = total_bytes * pct / 100ULL;
  return std::max(8ULL * 1024ULL * 1024ULL * 1024ULL, pct_bytes);
}

void TrimDeviceMemoryPoolIfRequested() {
  if (!TrimDeviceMemoryAfterEval() || !g_scheme || !g_scheme->context) {
    return;
  }
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (status == cudaSuccess && total_bytes > 0) {
    const unsigned long long free_u64 =
        static_cast<unsigned long long>(free_bytes);
    const unsigned long long total_u64 =
        static_cast<unsigned long long>(total_bytes);
    if (free_u64 >= DeviceMemoryTrimMinFreeBytes(total_u64)) {
      return;
    }
  }
  const auto started = std::chrono::steady_clock::now();
  g_scheme->context->memory_pool_.TrimTo(
      static_cast<std::size_t>(
          DeviceMemoryTrimDefaultTargetBytes(
              static_cast<unsigned long long>(total_bytes))));
  const auto finished = std::chrono::steady_clock::now();
  g_device_memory_trim_seconds +=
      std::chrono::duration<double>(finished - started).count();
}

void EnsureRotationKeyPrepared(int key, int level) {
  if (key <= 0) {
    return;
  }
  if (level < 0 || level > g_scheme->param->max_level_) {
    AbortWithMessage("rotation key level is outside supported preset range");
  }
  auto prepared_it = g_scheme->prepared_rotation_key_levels.find(key);
  if (prepared_it != g_scheme->prepared_rotation_key_levels.end() &&
      prepared_it->second == level) {
    return;
  }
  if (prepared_it != g_scheme->prepared_rotation_key_levels.end()) {
    g_scheme->interface->RemoveRotationKeys();
    g_scheme->prepared_rotation_key_levels.clear();
  }
  g_scheme->interface->PrepareRotationKey(key, level);
  g_scheme->prepared_rotation_key_levels[key] = level;
}

const EvkMap &LinearTransformEvkMap(const LinearTransformState &state) {
  if (state.rotation_keys) {
    return *state.rotation_keys;
  }
  return g_scheme->interface->GetEvkMap();
}

bool LinearTransformHasRotationKey(const LinearTransformState &state, int key) {
  return state.rotation_keys && state.rotation_keys->find(key) != state.rotation_keys->end();
}

void SetLinearTransformRotationKey(LinearTransformState &state, int key,
                                   Evk &&evk) {
  if (key <= 0) {
    return;
  }
  if (!state.rotation_keys) {
    state.rotation_keys = std::make_unique<EvkMap>();
  }
  state.rotation_keys->erase(key);
  state.rotation_keys->try_emplace(key, std::move(evk));
}

void RemoveLinearTransformRotationKeysState(LinearTransformState &state) {
  state.rotation_keys.reset();
}

void EnsureBootstrapPrepared(int num_slots) {
  if (!g_scheme->bootstrap_interface) {
    SyncBootstrapInterfaceSecrets(*g_scheme);
  }
  if (!g_scheme->eval_mod_prepared) {
    g_scheme->context->PrepareEvalMod();
    g_scheme->eval_mod_prepared = true;
  }
  const bool prepare_slot = g_scheme->prepared_boot_slots.insert(num_slots).second;
  if (prepare_slot) {
    g_scheme->context->PrepareEvalSpecialFFT(num_slots);
    cheddar::EvkRequest req;
    g_scheme->context->AddRequiredRotations(req, num_slots);
    g_scheme->bootstrap_interface->PrepareRotationKey(req);
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

ArrayResultUInt64 GetDeviceMemoryInfo() {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (status != cudaSuccess) {
    return MakeUInt64ArrayResult({0, 0});
  }
  return MakeUInt64ArrayResult({
      static_cast<unsigned long long>(free_bytes),
      static_cast<unsigned long long>(total_bytes),
  });
}

void SynchronizeDevice() {
  const cudaError_t status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    AbortWithMessage("cudaDeviceSynchronize failed");
  }
}

void TrimDeviceMemoryPool(unsigned long long targetBytes) {
  RequireScheme();
  const auto started = std::chrono::steady_clock::now();
  g_scheme->context->memory_pool_.TrimTo(static_cast<std::size_t>(targetBytes));
  const auto finished = std::chrono::steady_clock::now();
  g_device_memory_trim_seconds +=
      std::chrono::duration<double>(finished - started).count();
}

double ConsumeDeviceMemoryTrimSeconds() {
  const double value = g_device_memory_trim_seconds;
  g_device_memory_trim_seconds = 0.0;
  return value;
}

ArrayResultDouble ConsumeSharedCacheEvalProfileSeconds() {
  std::vector<double> values = {
      g_shared_cache_eval_profile.plan_s,
      g_shared_cache_eval_profile.level_adjust_s,
      g_shared_cache_eval_profile.baby_step_s,
      g_shared_cache_eval_profile.giant_step_s,
      g_shared_cache_eval_profile.stream_build_map_s,
      g_shared_cache_eval_profile.stream_encode_hoist_s,
      g_shared_cache_eval_profile.stream_load_payload_s,
      g_shared_cache_eval_profile.stream_eval_s,
      g_shared_cache_eval_profile.stream_accumulate_s,
      g_shared_cache_eval_profile.push_s,
      g_shared_cache_eval_profile.trim_s,
  };
  g_shared_cache_eval_profile = SharedCacheEvalProfile();
  return MakeDoubleArrayResult(values);
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
  const int normalized = NormalizeRotationIndex(rotation, DefaultRotationWidth());
  EnsureRotationKeyPrepared(normalized, g_scheme->param->default_encryption_level_);
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
  const int normalized = NormalizeCiphertextRotation(ciphertextID, amount);
  EnsureRotationKeyPrepared(normalized,
                            CiphertextLevel(RetrieveCiphertext(ciphertextID)));
  return ApplyCiphertextUnaryInPlace(ciphertextID, [&](Ct &out, const Ct &in) {
    g_scheme->context->HRot(out, in,
                            g_scheme->interface->GetRotationKey(normalized),
                            normalized);
  });
}

int RotateNew(int ciphertextID, int amount) {
  RequireScheme();
  const int normalized = NormalizeCiphertextRotation(ciphertextID, amount);
  EnsureRotationKeyPrepared(normalized,
                            CiphertextLevel(RetrieveCiphertext(ciphertextID)));
  return ApplyCiphertextUnaryNew(ciphertextID, [&](Ct &out, const Ct &in) {
    g_scheme->context->HRot(out, in,
                            g_scheme->interface->GetRotationKey(normalized),
                            normalized);
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
  const double scale = RetrieveCiphertext(ciphertextID).GetScale();
  return ApplyCiphertextConstInPlace(
      ciphertextID, static_cast<double>(scalar), scale,
      [&](Ct &out, const Ct &in, const Const &constant) {
        g_scheme->context->Add(out, in, constant);
      });
}

int AddScalarNew(int ciphertextID, float scalar) {
  RequireScheme();
  const double scale = RetrieveCiphertext(ciphertextID).GetScale();
  return ApplyCiphertextConstNew(
      ciphertextID, static_cast<double>(scalar), scale,
      [&](Ct &out, const Ct &in, const Const &constant) {
        g_scheme->context->Add(out, in, constant);
      });
}

int SubScalar(int ciphertextID, float scalar) {
  RequireScheme();
  const double scale = RetrieveCiphertext(ciphertextID).GetScale();
  return ApplyCiphertextConstInPlace(
      ciphertextID, static_cast<double>(scalar), scale,
      [&](Ct &out, const Ct &in, const Const &constant) {
        g_scheme->context->Sub(out, in, constant);
      });
}

int SubScalarNew(int ciphertextID, float scalar) {
  RequireScheme();
  const double scale = RetrieveCiphertext(ciphertextID).GetScale();
  return ApplyCiphertextConstNew(
      ciphertextID, static_cast<double>(scalar), scale,
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
                            float bsgsRatio, const char *ioMode) {
  RequireScheme();
  if ((ioMode != nullptr && std::strcmp(ioMode, "load") == 0) ||
      diagDataLen == 0) {
    return AddLinearTransformFromDescriptor(
        diagIdxs, diagIdxsLen, DefaultRotationWidth(), level, bsgsRatio);
  }
  return AddLinearTransformFromMatrix(
      BuildRealStripedMatrix(diagIdxs, diagIdxsLen, diagData, diagDataLen),
      level, bsgsRatio);
}

ArrayResultInt GenerateLinearTransformsBatch(
    int numTransforms, const int *const *diagIdxsArray, const int *diagIdxsLens,
    const float *const *diagDataArray, const int *diagDataLens,
    const int *levels, float bsgsRatio, const char *ioMode) {
  RequireScheme();
  std::vector<int> ids;
  ids.reserve(std::max(0, numTransforms));
  const bool descriptor_only = ioMode != nullptr && std::strcmp(ioMode, "load") == 0;
  for (int i = 0; i < numTransforms; ++i) {
    if (descriptor_only || diagDataLens[i] == 0) {
      ids.push_back(AddLinearTransformFromDescriptor(
          diagIdxsArray[i], diagIdxsLens[i], DefaultRotationWidth(), levels[i],
          bsgsRatio));
    } else {
      ids.push_back(AddLinearTransformFromMatrix(
          BuildRealStripedMatrix(diagIdxsArray[i], diagIdxsLens[i],
                                 diagDataArray[i], diagDataLens[i]),
          levels[i], bsgsRatio));
    }
  }
  return MakeIntArrayResult(ids);
}

int EvaluateLinearTransform(int transformID, int ciphertextID) {
  RequireScheme();
  Ct output;
  LinearTransformState &state = RetrieveTransform(transformID);
  if (state.singleton) {
    const int output_id = EvaluateSingletonLinearTransform(state, ciphertextID);
    TrimDeviceMemoryPoolIfRequested();
    return output_id;
  }
  if (ShouldStreamLinearTransform(state)) {
    const int output_id = EvaluateLinearTransformStreaming(state, ciphertextID);
    TrimDeviceMemoryPoolIfRequested();
    return output_id;
  }
  LinearTransform &transform = EnsureTransformLoaded(state);
  const Ct &input = RetrieveCiphertext(ciphertextID);
  const int input_level = CiphertextLevel(input);
  if (input_level < state.level) {
    AbortWithMessage("input ciphertext level is below the linear transform level");
  }
  Ct leveled_input;
  const Ct &eval_input = CiphertextAtLevel(leveled_input, input, state.level);
  transform.Evaluate(g_scheme->context, output, eval_input,
                     LinearTransformEvkMap(state),
                     kUseMinKSLinearTransforms);
  const int output_id = PushCiphertext(std::move(output));
  TrimDeviceMemoryPoolIfRequested();
  return output_id;
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
  LinearTransformState &state = RetrieveTransform(transformID);
  if (state.metadata_cached && state.matrix.empty()) {
    std::vector<int> keys;
    keys.reserve(state.cached_rotation_key_requests.size());
    for (const auto &[key, _level] : state.cached_rotation_key_requests) {
      if (key != 0) {
        keys.push_back(key);
      }
    }
    return MakeIntArrayResult(keys);
  }
  if (state.singleton) {
    const int key = SingletonLinearTransformRotationKey(state);
    std::vector<int> keys;
    if (key != 0) {
      keys.push_back(key);
    }
    return MakeIntArrayResult(keys);
  }
  cheddar::EvkRequest req;
  AddLinearTransformRequiredRotations(state, req);
  std::vector<int> keys;
  keys.reserve(req.size());
  for (const auto &[key, _] : req) {
    if (key != 0) {
      keys.push_back(key);
    }
  }
  return MakeIntArrayResult(keys);
}

ArrayResultInt GetLinearTransformRotationKeyRequests(int transformID) {
  RequireScheme();
  LinearTransformState &state = RetrieveTransform(transformID);
  if (state.metadata_cached && state.matrix.empty()) {
    std::vector<int> flat;
    flat.reserve(state.cached_rotation_key_requests.size() * 2);
    for (const auto &[key, level] : state.cached_rotation_key_requests) {
      if (key != 0) {
        flat.push_back(key);
        flat.push_back(level);
      }
    }
    return MakeIntArrayResult(flat);
  }
  if (state.singleton) {
    const int key = SingletonLinearTransformRotationKey(state);
    std::vector<int> flat;
    if (key != 0) {
      flat.push_back(key);
      flat.push_back(state.level);
    }
    return MakeIntArrayResult(flat);
  }
  cheddar::EvkRequest req;
  AddLinearTransformRequiredRotations(state, req);
  std::vector<int> flat;
  flat.reserve(req.size() * 2);
  for (const auto &[key, level] : req) {
    if (key != 0) {
      flat.push_back(key);
      flat.push_back(level);
    }
  }
  return MakeIntArrayResult(flat);
}

ArrayResultUInt64 EstimateLinearTransformDeviceBytes(int transformID) {
  RequireScheme();
  const LinearTransformState &state = RetrieveTransform(transformID);
  if (state.metadata_cached && state.matrix.empty()) {
    return MakeUInt64ArrayResult({state.cached_device_bytes});
  }
  if (state.singleton) {
    return MakeUInt64ArrayResult(
        {static_cast<unsigned long long>(state.singleton_values.size() *
                                         sizeof(Complex))});
  }
  return MakeUInt64ArrayResult({EstimateLinearTransformStateDeviceBytes(state)});
}

int LinearTransformUsesStreaming(int transformID) {
  RequireScheme();
  LinearTransformState &state = RetrieveTransform(transformID);
  if (state.metadata_cached && state.matrix.empty()) {
    return state.cached_uses_streaming;
  }
  return ShouldStreamLinearTransform(state) ? 1 : 0;
}

void GenerateLinearTransformRotationKey(int key) {
  RequireScheme();
  EnsureRotationKeyPrepared(key, g_scheme->param->default_encryption_level_);
}

void GenerateLinearTransformRotationKeyAtLevel(int key, int level) {
  RequireScheme();
  EnsureRotationKeyPrepared(key, level);
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
        levels[i], 2.0f));
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
        levels[i], 2.0f));
  }
  return MakeIntArrayResult(ids);
}

ArrayResultInt GenerateLinearTransformsUnifiedLoad(
    int numTransforms, const int *const *diagIdxsArray, const int *diagIdxsLens,
    const int *levels) {
  RequireScheme();
  std::vector<int> ids;
  ids.reserve(numTransforms);
  const int width = DefaultLinearTransformDescriptorWidth();
  for (int i = 0; i < numTransforms; ++i) {
    ids.push_back(AddLinearTransformFromDescriptor(
        diagIdxsArray[i], diagIdxsLens[i], width, levels[i], 2.0f));
  }
  return MakeIntArrayResult(ids);
}

ArrayResultInt EvaluateLinearTransformsWithSharedCache(const int *transformIDs,
                                                       int numTransforms,
                                                       int ciphertextID) {
  RequireScheme();
  SharedCacheEvalProfile profile;
  std::vector<int> ordered_ids;
  ordered_ids.reserve(numTransforms);
  for (int i = 0; i < numTransforms; ++i) {
    ordered_ids.push_back(transformIDs[i]);
  }
  if (HasSingletonLinearTransform(ordered_ids)) {
    std::vector<int> output_ids;
    output_ids.reserve(numTransforms);
    const auto eval_started = std::chrono::steady_clock::now();
    for (int transform_id : ordered_ids) {
      output_ids.push_back(EvaluateLinearTransform(transform_id, ciphertextID));
    }
    AddDuration(profile.giant_step_s, eval_started);
    AccumulateSharedCacheEvalProfile(profile);
    return MakeIntArrayResult(output_ids);
  }
  SharedCachePlan local_plan;
  SharedCachePlan *plan_ptr = nullptr;
  const auto plan_started = std::chrono::steady_clock::now();
  const Ct &input = RetrieveCiphertext(ciphertextID);
  const int input_level = CiphertextLevel(input);
  bool needs_level_limited_plan = false;
  for (int transform_id : ordered_ids) {
    if (RetrieveTransform(transform_id).level > input_level) {
      needs_level_limited_plan = true;
      break;
    }
  }
  if (PersistSharedCachePlans() && !needs_level_limited_plan) {
    plan_ptr = &GetOrBuildSharedCachePlan(ordered_ids);
  } else {
    local_plan = BuildSharedCachePlan(
        ordered_ids, needs_level_limited_plan ? input_level : -1);
    plan_ptr = &local_plan;
  }
  AddDuration(profile.plan_s, plan_started);
  SharedCachePlan &plan = *plan_ptr;
  std::map<int, int> output_id_by_transform;

  for (const SharedCacheBucket &bucket : plan.buckets) {
    const Ct *bucket_input = &input;
    Ct leveled_input;
    std::unique_ptr<HoistHandler> lowered_cache;
    HoistHandler *bucket_cache = bucket.cache.get();
    const auto level_started = std::chrono::steady_clock::now();
    const int eval_level = std::min(input_level, bucket.level);
    if (input_level > eval_level) {
      bucket_input = &CiphertextAtLevel(leveled_input, input, eval_level);
    }
    if (eval_level != bucket.level) {
      for (int transform_id : bucket.transform_ids) {
        PrepareLinearTransformRotationKeysAtLevel(
            RetrieveTransform(transform_id), eval_level);
      }
      lowered_cache = BuildSharedCacheHoist(
          eval_level, SharedCacheBabyStepsForLevel(bucket, eval_level));
      bucket_cache = lowered_cache.get();
    }
    AddDuration(profile.level_adjust_s, level_started);

    const EvkMap &bucket_evk_map =
        LinearTransformEvkMap(RetrieveTransform(bucket.transform_ids.front()));
    std::map<int, Ct> bs_cache;
    const auto baby_started = std::chrono::steady_clock::now();
    bucket_cache->EvaluateBabyStep(g_scheme->context, bs_cache, *bucket_input,
                                   bucket_evk_map, false);
    AddDuration(profile.baby_step_s, baby_started);

    for (int transform_id : bucket.transform_ids) {
      Ct output;
      LinearTransformState &transform_state = RetrieveTransform(transform_id);
      std::unique_ptr<LinearTransformState> lowered_transform_state;
      LinearTransformState *eval_transform_state = &transform_state;
      if (eval_level != transform_state.level) {
        lowered_transform_state =
            std::make_unique<LinearTransformState>(
                MakeLinearTransformLevelView(transform_state, eval_level));
        eval_transform_state = lowered_transform_state.get();
      }
      const auto giant_started = std::chrono::steady_clock::now();
      if (ShouldStreamLinearTransform(*eval_transform_state)) {
        EvaluateLinearTransformGiantStepStreaming(
            *eval_transform_state, output, bs_cache,
            LinearTransformEvkMap(transform_state), &profile);
      } else {
        LinearTransform &transform = EnsureTransformLoaded(*eval_transform_state);
        transform.EvaluateGiantStep(g_scheme->context, output, bs_cache,
                                    LinearTransformEvkMap(transform_state),
                                    kUseMinKSLinearTransforms);
      }
      AddDuration(profile.giant_step_s, giant_started);
      const auto push_started = std::chrono::steady_clock::now();
      output_id_by_transform[transform_id] = PushCiphertext(std::move(output));
      AddDuration(profile.push_s, push_started);
    }
  }

  std::vector<int> output_ids;
  output_ids.reserve(numTransforms);
  for (int i = 0; i < numTransforms; ++i) {
    output_ids.push_back(output_id_by_transform.at(transformIDs[i]));
  }
  const double trim_before = g_device_memory_trim_seconds;
  TrimDeviceMemoryPoolIfRequested();
  profile.trim_s += g_device_memory_trim_seconds - trim_before;
  AccumulateSharedCacheEvalProfile(profile);
  return MakeIntArrayResult(output_ids);
}

void PrepareLinearTransformsSharedCachePlan(const int *transformIDs,
                                            int numTransforms) {
  RequireScheme();
  if (numTransforms <= 1) {
    return;
  }
  std::vector<int> ordered_ids;
  ordered_ids.reserve(numTransforms);
  for (int i = 0; i < numTransforms; ++i) {
    ordered_ids.push_back(transformIDs[i]);
  }
  (void)GetOrBuildSharedCachePlan(ordered_ids);
}

ArrayResultByte GenerateAndSerializeRotationKey(int key) {
  RequireScheme();
  EnsureRotationKeyPrepared(key, g_scheme->param->default_encryption_level_);
  const auto bytes =
      SerializeEvaluationKeyBytes(g_scheme->interface->GetRotationKey(key));
  g_scheme->interface->RemoveRotationKeys();
  g_scheme->prepared_rotation_key_levels.clear();
  return MakeByteArrayResult(bytes);
}

ArrayResultByte GenerateAndSerializeRotationKeyAtLevel(int key, int level) {
  RequireScheme();
  EnsureRotationKeyPrepared(key, level);
  const auto bytes =
      SerializeEvaluationKeyBytes(g_scheme->interface->GetRotationKey(key));
  g_scheme->interface->RemoveRotationKeys();
  g_scheme->prepared_rotation_key_levels.clear();
  return MakeByteArrayResult(bytes);
}

void LoadRotationKey(const unsigned char *data, unsigned long lenData,
                     unsigned long key) {
  RequireScheme();
  const int key_int = static_cast<int>(key);
  const int level =
      SerializedEvaluationKeyLevel(data, static_cast<std::size_t>(lenData));
  const auto prepared_it = g_scheme->prepared_rotation_key_levels.find(key_int);
  if (prepared_it != g_scheme->prepared_rotation_key_levels.end() &&
      prepared_it->second == level) {
    return;
  }
  if (prepared_it != g_scheme->prepared_rotation_key_levels.end()) {
    g_scheme->interface->RemoveRotationKeys();
    g_scheme->prepared_rotation_key_levels.clear();
  }
  auto evk =
      DeserializeEvaluationKeyBytes(data, static_cast<std::size_t>(lenData));
  g_scheme->interface->SetRotationKey(key_int, std::move(evk));
  g_scheme->prepared_rotation_key_levels[key_int] = level;
}

void LoadLinearTransformRotationKey(const unsigned char *data,
                                    unsigned long lenData,
                                    unsigned long key,
                                    int transformID) {
  RequireScheme();
  auto evk =
      DeserializeEvaluationKeyBytes(data, static_cast<std::size_t>(lenData));
  SetLinearTransformRotationKey(
      RetrieveTransform(transformID), static_cast<int>(key), std::move(evk));
}

void RemoveLinearTransformRotationKeys(int transformID) {
  RequireScheme();
  RemoveLinearTransformRotationKeysState(RetrieveTransform(transformID));
}

ArrayResultByte SerializeDiagonal(int transformID, int diagIdx) {
  RequireScheme();
  LinearTransformState &state = RetrieveTransform(transformID);
  const auto bytes = SerializeDiagonalBytes(state, diagIdx);
  state.transform.reset();
  if (!PersistSharedCachePlans()) {
    g_scheme->shared_cache_plans.clear();
  }
  return MakeByteArrayResult(bytes);
}

ArrayResultByte SerializeLinearTransformPlaintexts(int transformID) {
  RequireScheme();
  LinearTransformState &state = RetrieveTransform(transformID);
  const auto bytes = SerializeLinearTransformPlaintextsBytes(state);
  state.transform.reset();
  if (!PersistSharedCachePlans()) {
    g_scheme->shared_cache_plans.clear();
  }
  return MakeByteArrayResult(bytes);
}

void LoadPlaintextDiagonal(const unsigned char * /*data*/,
                           unsigned long lenData, int transformID,
                           unsigned long /*diagIdx*/) {
  RequireScheme();
  LinearTransformState &state = RetrieveTransform(transformID);
  if (state.singleton) {
    return;
  }
  if (lenData == 0) {
    if (!ShouldStreamLinearTransform(state)) {
      EnsureTransformLoaded(state);
    }
    return;
  }
  EnsureTransformLoaded(state);
}

void LoadPlaintextDiagonalsBatch(const unsigned char * /*data*/,
                                 unsigned long lenData,
                                 const unsigned long long * /*offsets*/,
                                 int /*numOffsets*/,
                                 const unsigned long long * /*lengths*/,
                                 int /*numLengths*/,
                                 const int * /*diagIdxs*/,
                                 int /*numDiagIdxs*/,
                                 int transformID) {
  RequireScheme();
  LinearTransformState &state = RetrieveTransform(transformID);
  if (state.singleton) {
    return;
  }
  if (lenData == 0) {
    if (!ShouldStreamLinearTransform(state)) {
      EnsureTransformLoaded(state);
    }
    return;
  }
  EnsureTransformLoaded(state);
}

void LoadLinearTransformPlaintexts(const unsigned char *data,
                                   unsigned long lenData,
                                   int transformID) {
  RequireScheme();
  LinearTransformState &state = RetrieveTransform(transformID);
  if (state.singleton) {
    return;
  }
  if (lenData == 0) {
    return;
  }
  LoadLinearTransformPlaintextsBytes(
      state, data, static_cast<std::size_t>(lenData));
}

void RemovePlaintextDiagonals(int transformID) {
  RequireScheme();
  RetrieveTransform(transformID).transform.reset();
  if (!PersistSharedCachePlans()) {
    g_scheme->shared_cache_plans.clear();
  }
}

void ReleaseLinearTransformMatrix(int transformID) {
  RequireScheme();
  ReleaseLinearTransformMatrixState(RetrieveTransform(transformID));
  if (!PersistSharedCachePlans()) {
    g_scheme->shared_cache_plans.clear();
  }
}

void RemoveRotationKeys() {
  RequireScheme();
  g_scheme->interface->RemoveRotationKeys();
  g_scheme->prepared_rotation_key_levels.clear();
}

void NewBootstrapper(const int * /*logPs*/, int /*lenLogPs*/, int numSlots) {
  RequireScheme();
  EnsureBootstrapPrepared(numSlots);
}

int Bootstrap(int ciphertextID, int numSlots) {
  RequireScheme();
  EnsureBootstrapPrepared(numSlots);
  const Ct &input = RetrieveCiphertext(ciphertextID);
  if (numSlots <= 0 || numSlots > input.GetNumSlots()) {
    AbortWithMessage("bootstrap slot count is outside ciphertext slot range");
  }
  Ct boot_input;
  g_scheme->context->Copy(boot_input, input);
  // Orion chooses the sparse bootstrap width at compile time. Cheddar's
  // BootContext derives the FFT plan from ciphertext metadata, so align this
  // temporary copy with the requested sparse slot count before bootstrapping.
  boot_input.SetNumSlots(numSlots);
  Ct output;
  g_scheme->context->Boot(output, boot_input,
                          g_scheme->bootstrap_interface->GetEvkMap());
  const int half_degree = g_scheme->param->degree_ / 2;
  if (numSlots < half_degree) {
    Const sparse_compensation =
        EncodeScalarConstant(CiphertextLevel(output), 1.0,
                             static_cast<double>(half_degree) /
                                 static_cast<double>(numSlots));
    Ct compensated;
    g_scheme->context->Mult(compensated, output, sparse_compensation);
    output = std::move(compensated);
  }
  const int output_id = PushCiphertext(std::move(output));
  TrimDeviceMemoryPoolIfRequested();
  return output_id;
}

void DeleteBootstrappers() {
  if (!g_scheme) {
    return;
  }
  g_scheme->prepared_boot_slots.clear();
}

}  // extern "C"
