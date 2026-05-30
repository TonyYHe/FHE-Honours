#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

extern "C" {

struct OrionDiagPayload {
  int row;
  int col;
  int level;
  const char *task_id;
  int *diag_indices;
  unsigned long diag_indices_len;
  float *diag_data;
  unsigned long diag_data_len;
};

struct OrionDiagPayloadBatch {
  OrionDiagPayload *payloads;
  unsigned long len;
  int output_rotations;
  const char *builder_kind;
  const char *fallback_reason;
};

struct OrionProviderNativeSourceSpec {
  int slots;
  int c_in;
  int h_in;
  int w_in;
  int c_out;
  int h_out;
  int w_out;
  int gap_in;
  int gap_out;
  int kernel;
  int stride;
  int pad;
  int dilation;
  int input_h_min;
  int input_h_max;
  int output_top_beta;
  int output_bottom_beta;
  int output_physical_top_beta;
  int output_physical_bottom_beta;
  int stripe_index;
  int stripe_source_h_start;
  int stripe_source_h;
  int stripe_target_h_start;
  int stripe_target_h_end;
  int stripe_target_h;
  int source_tile;
  int target_tile;
  int source_group;
  int target_group;
  int source_index;
  int target_index;
  int compact_output;
  int compact_target_block;
};

struct OrionProviderCompactSourceSpec {
  int slots;
  int c_in;
  int h_in;
  int w_in;
  int c_out;
  int h_out;
  int w_out;
  int gap_out;
  int kernel;
  int stride;
  int pad;
  int dilation;
  int source_top_beta;
  int source_bottom_beta;
  int source_gap;
  int output_top_beta;
  int output_bottom_beta;
  int output_physical_top_beta;
  int output_physical_bottom_beta;
  int stripe_index;
  int stripe_target_h_start;
  int stripe_target_h_end;
  int stripe_target_h;
  int target_tile;
  int source_block;
  int target_group;
  int target_index;
  int compact_output;
  int compact_target_block;
};

}

namespace {

constexpr const char *kBuilderKind = "cpp_dense_conv2d";

std::string g_last_error;

template <typename T>
T *AllocArray(std::size_t count) {
  if (count == 0) {
    return nullptr;
  }
  void *ptr = std::malloc(sizeof(T) * count);
  if (ptr == nullptr) {
    throw std::bad_alloc();
  }
  return static_cast<T *>(ptr);
}

char *CopyCString(const std::string &value) {
  char *out = AllocArray<char>(value.size() + 1);
  std::memcpy(out, value.c_str(), value.size() + 1);
  return out;
}

int CeilDiv(int64_t value, int64_t divisor) {
  if (divisor <= 0) {
    throw std::invalid_argument("invalid divisor");
  }
  return static_cast<int>((value + divisor - 1) / divisor);
}

int64_t Numel4(const int *shape) {
  return static_cast<int64_t>(shape[0]) * static_cast<int64_t>(shape[1]) *
         static_cast<int64_t>(shape[2]) * static_cast<int64_t>(shape[3]);
}

int64_t PackedFlatIndex(
    int channel,
    int64_t row,
    int64_t col,
    int gap,
    int height,
    int width,
    int row_offset) {
  const int phases = gap * gap;
  const int phase = channel % phases;
  const int packed_channel = channel / phases;
  const int64_t packed_row = row * gap + phase / gap + row_offset;
  const int64_t packed_col = col * gap + phase % gap;
  return (static_cast<int64_t>(packed_channel) * height + packed_row) * width + packed_col;
}

int64_t GapChannelPosition(
    int channel,
    int64_t h,
    int64_t w,
    int height,
    int width,
    int gap) {
  const int g = std::max(1, gap);
  const int phases = g * g;
  const int packed_w = width * g;
  const int group_block = height * g * packed_w;
  const int group = channel / phases;
  const int phase = channel % phases;
  const int phase_h = phase / g;
  const int phase_w = phase % g;
  return static_cast<int64_t>(group) * group_block + (h * g + phase_h) * packed_w + w * g + phase_w;
}

int64_t MaterializedOutputSourceH(int64_t output_h, int h_out, int output_top_beta, int output_bottom_beta) {
  int64_t value = output_h;
  if (value < 0) {
    value += output_top_beta;
  } else if (value >= h_out) {
    value -= output_bottom_beta;
  }
  return std::min<int64_t>(std::max<int64_t>(value, 0), std::max(0, h_out - 1));
}

int PhysicalOutputH(const OrionProviderNativeSourceSpec &spec) {
  return spec.h_out + std::max(0, spec.output_physical_top_beta) + std::max(0, spec.output_physical_bottom_beta);
}

bool PhysicalOutputHValid(const OrionProviderNativeSourceSpec &spec, int64_t output_h) {
  const int top = std::max(0, spec.output_physical_top_beta);
  const int bottom = std::max(0, spec.output_physical_bottom_beta);
  return output_h >= -top && output_h < spec.h_out + bottom;
}

int64_t PhysicalOutputHPosition(const OrionProviderNativeSourceSpec &spec, int64_t output_h) {
  return output_h + std::max(0, spec.output_physical_top_beta);
}

struct DenseSpec {
  int slots = 0;
  std::string embed_method;
  bool is_last_layer = false;
  bool allow_hybrid = true;
  int input_shape[4] = {0, 0, 0, 0};
  int output_shape[4] = {0, 0, 0, 0};
  int fhe_input_shape[4] = {0, 0, 0, 0};
  int fhe_output_shape[4] = {0, 0, 0, 0};
  int input_gap = 1;
  int output_gap = 1;
  int input_row_offset = 0;
  int output_row_offset = 0;
  int kernel_h = 0;
  int kernel_w = 0;
  int stride_h = 1;
  int stride_w = 1;
  int pad_h = 0;
  int pad_w = 0;
  int dilation_h = 1;
  int dilation_w = 1;
  int output_top_beta = 0;
  int output_bottom_beta = 0;
  bool fuse_output_relayout = false;
};

struct Accumulator {
  int64_t matrix_height = 0;
  int64_t matrix_width = 0;
  int slots = 0;
  int num_block_rows = 0;
  int num_block_cols = 0;
  int block_height = 0;
  int output_rotations = 0;
  bool restrict_blocks = false;
  std::vector<std::pair<int, int>> requested_blocks;
  std::map<std::pair<int, int>, std::map<int, std::vector<float>>> diagonals;

  bool BlockAllowed(int row, int col) const {
    if (!restrict_blocks) {
      return true;
    }
    return std::find(requested_blocks.begin(), requested_blocks.end(), std::make_pair(row, col)) != requested_blocks.end();
  }

  void AddEntry(int64_t row, int64_t col, float value) {
    if (value == 0.0f || row < 0 || row >= matrix_height || col < 0 || col >= matrix_width) {
      return;
    }

    int block_row = 0;
    int local_row = 0;
    int block_col = 0;
    int local_col = 0;
    int diag_idx = 0;
    int position = 0;
    if (block_height == slots) {
      block_row = static_cast<int>(row / slots);
      local_row = static_cast<int>(row - static_cast<int64_t>(block_row) * slots);
      block_col = static_cast<int>(col / slots);
      local_col = static_cast<int>(col - static_cast<int64_t>(block_col) * slots);
      diag_idx = (local_col - local_row) % slots;
      if (diag_idx < 0) {
        diag_idx += slots;
      }
      position = local_row;
    } else {
      block_row = 0;
      local_row = static_cast<int>(row);
      block_col = static_cast<int>(col / slots);
      local_col = static_cast<int>(col - static_cast<int64_t>(block_col) * slots);
      diag_idx = (local_col - local_row) % block_height;
      if (diag_idx < 0) {
        diag_idx += block_height;
      }
      position = (local_col - diag_idx) % slots;
      if (position < 0) {
        position += slots;
      }
    }

    if (!BlockAllowed(block_row, block_col)) {
      return;
    }
    auto &block = diagonals[std::make_pair(block_row, block_col)];
    auto it = block.find(diag_idx);
    if (it == block.end()) {
      it = block.emplace(diag_idx, std::vector<float>(static_cast<std::size_t>(slots), 0.0f)).first;
    }
    it->second[static_cast<std::size_t>(position)] += value;
  }
};

struct IndexAccumulator {
  int64_t matrix_height = 0;
  int64_t matrix_width = 0;
  int slots = 0;
  int num_block_rows = 0;
  int num_block_cols = 0;
  int block_height = 0;
  int output_rotations = 0;
  std::map<std::pair<int, int>, std::vector<int>> indices;

  void AddEntry(int64_t row, int64_t col, float value) {
    if (value == 0.0f || row < 0 || row >= matrix_height || col < 0 || col >= matrix_width) {
      return;
    }

    int block_row = 0;
    int block_col = 0;
    int diag_idx = 0;
    if (block_height == slots) {
      block_row = static_cast<int>(row / slots);
      const int local_row = static_cast<int>(row - static_cast<int64_t>(block_row) * slots);
      block_col = static_cast<int>(col / slots);
      const int local_col = static_cast<int>(col - static_cast<int64_t>(block_col) * slots);
      diag_idx = (local_col - local_row) % slots;
      if (diag_idx < 0) {
        diag_idx += slots;
      }
    } else {
      block_row = 0;
      block_col = static_cast<int>(col / slots);
      const int local_col = static_cast<int>(col - static_cast<int64_t>(block_col) * slots);
      diag_idx = (local_col - static_cast<int>(row)) % block_height;
      if (diag_idx < 0) {
        diag_idx += block_height;
      }
    }

    auto &block = indices[std::make_pair(block_row, block_col)];
    if (std::find(block.begin(), block.end(), diag_idx) == block.end()) {
      block.push_back(diag_idx);
    }
  }
};

Accumulator MakeAccumulator(const DenseSpec &spec, const std::vector<std::pair<int, int>> &blocks) {
  Accumulator acc;
  acc.matrix_height = Numel4(spec.fhe_output_shape);
  acc.matrix_width = Numel4(spec.fhe_input_shape);
  acc.slots = spec.slots;
  acc.num_block_rows = CeilDiv(acc.matrix_height, spec.slots);
  acc.num_block_cols = CeilDiv(acc.matrix_width, spec.slots);
  if (spec.allow_hybrid && acc.num_block_rows == 1 && spec.embed_method == "hybrid" && !spec.is_last_layer) {
    int block_height = 1;
    while (block_height < acc.matrix_height) {
      block_height <<= 1;
    }
    acc.block_height = block_height;
    acc.output_rotations = static_cast<int>(std::log2(static_cast<double>(spec.slots / block_height)));
  } else {
    acc.block_height = spec.slots;
    acc.output_rotations = 0;
  }
  acc.restrict_blocks = !blocks.empty();
  acc.requested_blocks = blocks;
  return acc;
}

IndexAccumulator MakeIndexAccumulator(const DenseSpec &spec) {
  IndexAccumulator acc;
  acc.matrix_height = Numel4(spec.fhe_output_shape);
  acc.matrix_width = Numel4(spec.fhe_input_shape);
  acc.slots = spec.slots;
  acc.num_block_rows = CeilDiv(acc.matrix_height, spec.slots);
  acc.num_block_cols = CeilDiv(acc.matrix_width, spec.slots);
  if (spec.allow_hybrid && acc.num_block_rows == 1 && spec.embed_method == "hybrid" && !spec.is_last_layer) {
    int block_height = 1;
    while (block_height < acc.matrix_height) {
      block_height <<= 1;
    }
    acc.block_height = block_height;
    acc.output_rotations = static_cast<int>(std::log2(static_cast<double>(spec.slots / block_height)));
  } else {
    acc.block_height = spec.slots;
    acc.output_rotations = 0;
  }
  return acc;
}

std::vector<std::pair<int, int>> RequestedBlocks(const int *block_rows, const int *block_cols, int block_count) {
  std::vector<std::pair<int, int>> blocks;
  if (block_count <= 0) {
    return blocks;
  }
  if (block_rows == nullptr || block_cols == nullptr) {
    throw std::invalid_argument("block rows/cols are required when block_count > 0");
  }
  blocks.reserve(static_cast<std::size_t>(block_count));
  for (int i = 0; i < block_count; ++i) {
    blocks.emplace_back(block_rows[i], block_cols[i]);
  }
  std::sort(blocks.begin(), blocks.end());
  blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());
  return blocks;
}

void ValidateSpec(const DenseSpec &spec, int weight_len) {
  if (spec.slots <= 0) {
    throw std::invalid_argument("slots must be positive");
  }
  if (spec.input_gap <= 0 || spec.output_gap <= 0) {
    throw std::invalid_argument("gaps must be positive");
  }
  if (spec.kernel_h <= 0 || spec.kernel_w <= 0) {
    throw std::invalid_argument("kernel must be positive");
  }
  const int64_t expected_weight =
      static_cast<int64_t>(spec.output_shape[1]) * static_cast<int64_t>(spec.input_shape[1]) *
      spec.kernel_h * spec.kernel_w;
  if (expected_weight != weight_len) {
    throw std::invalid_argument("weight length does not match dense Conv2d spec");
  }
}

template <typename TAccumulator>
void FillDenseConv2D(const DenseSpec &spec, const float *weight, TAccumulator &acc) {
  const int n_batch = spec.input_shape[0];
  const int ci = spec.input_shape[1];
  const int hi = spec.input_shape[2];
  const int wi = spec.input_shape[3];
  const int co = spec.output_shape[1];
  const int ho = spec.output_shape[2];
  const int wo = spec.output_shape[3];
  const int on_ci = spec.fhe_input_shape[1];
  const int on_hi = spec.fhe_input_shape[2];
  const int on_wi = spec.fhe_input_shape[3];
  const int on_co = spec.fhe_output_shape[1];
  const int on_ho = spec.fhe_output_shape[2];
  const int on_wo = spec.fhe_output_shape[3];
  const int64_t input_block_size = static_cast<int64_t>(on_ci) * on_hi * on_wi;
  const int64_t output_block_size = static_cast<int64_t>(on_co) * on_ho * on_wo;

  for (int oc = 0; oc < co; ++oc) {
    for (int ic = 0; ic < ci; ++ic) {
      for (int kh = 0; kh < spec.kernel_h; ++kh) {
        for (int kw = 0; kw < spec.kernel_w; ++kw) {
          const int64_t weight_index =
              (((static_cast<int64_t>(oc) * ci + ic) * spec.kernel_h + kh) * spec.kernel_w + kw);
          const float coeff = weight[weight_index];
          if (coeff == 0.0f) {
            continue;
          }
          for (int target_oh = -spec.output_top_beta; target_oh < ho + spec.output_bottom_beta; ++target_oh) {
            int op_oh = target_oh;
            if (spec.fuse_output_relayout) {
              if (op_oh < 0) {
                op_oh += spec.output_top_beta;
              } else if (op_oh >= ho) {
                op_oh -= spec.output_bottom_beta;
              }
              op_oh = std::min(std::max(op_oh, 0), std::max(0, ho - 1));
            }
            const int64_t ih = static_cast<int64_t>(op_oh) * spec.stride_h - spec.pad_h + static_cast<int64_t>(kh) * spec.dilation_h;
            if (ih < 0 || ih >= hi) {
              continue;
            }
            for (int ow = 0; ow < wo; ++ow) {
              const int64_t iw_value = static_cast<int64_t>(ow) * spec.stride_w - spec.pad_w + static_cast<int64_t>(kw) * spec.dilation_w;
              if (iw_value < 0 || iw_value >= wi) {
                continue;
              }
              const int64_t local_row = PackedFlatIndex(
                  oc,
                  target_oh,
                  ow,
                  spec.output_gap,
                  on_ho,
                  on_wo,
                  spec.output_row_offset);
              const int64_t local_col = PackedFlatIndex(
                  ic,
                  ih,
                  iw_value,
                  spec.input_gap,
                  on_hi,
                  on_wi,
                  spec.input_row_offset);
              for (int batch = 0; batch < n_batch; ++batch) {
                acc.AddEntry(
                    local_row + static_cast<int64_t>(batch) * output_block_size,
                    local_col + static_cast<int64_t>(batch) * input_block_size,
                    coeff);
              }
            }
          }
        }
      }
    }
  }
}

OrionDiagPayloadBatch BuildDensePayloadBatch(const DenseSpec &spec, const float *weight, int weight_len, const std::vector<std::pair<int, int>> &blocks) {
  ValidateSpec(spec, weight_len);
  Accumulator acc = MakeAccumulator(spec, blocks);
  FillDenseConv2D(spec, weight, acc);

  std::vector<std::pair<int, int>> block_keys;
  if (blocks.empty()) {
    for (int row = 0; row < acc.num_block_rows; ++row) {
      for (int col = 0; col < acc.num_block_cols; ++col) {
        block_keys.emplace_back(row, col);
      }
    }
  } else {
    for (const auto &block : blocks) {
      if (block.first >= 0 && block.first < acc.num_block_rows && block.second >= 0 && block.second < acc.num_block_cols) {
        block_keys.push_back(block);
      }
    }
  }
  std::sort(block_keys.begin(), block_keys.end());

  OrionDiagPayloadBatch out{nullptr, 0, acc.output_rotations, kBuilderKind, nullptr};
  if (block_keys.empty()) {
    return out;
  }
  out.payloads = AllocArray<OrionDiagPayload>(block_keys.size());
  out.len = static_cast<unsigned long>(block_keys.size());
  for (std::size_t i = 0; i < block_keys.size(); ++i) {
    const auto &block_key = block_keys[i];
    const auto block_it = acc.diagonals.find(block_key);
    const bool empty = block_it == acc.diagonals.end() || block_it->second.empty();
    const int diag_count = empty ? 1 : static_cast<int>(block_it->second.size());
    OrionDiagPayload payload{};
    payload.row = block_key.first;
    payload.col = block_key.second;
    payload.level = 0;
    payload.task_id = nullptr;
    payload.diag_indices = AllocArray<int>(static_cast<std::size_t>(diag_count));
    payload.diag_indices_len = static_cast<unsigned long>(diag_count);
    payload.diag_data = AllocArray<float>(static_cast<std::size_t>(diag_count) * static_cast<std::size_t>(spec.slots));
    payload.diag_data_len = static_cast<unsigned long>(diag_count * spec.slots);
    if (empty) {
      payload.diag_indices[0] = 0;
      std::fill(payload.diag_data, payload.diag_data + spec.slots, 0.0f);
    } else {
      int offset = 0;
      for (const auto &diag : block_it->second) {
        payload.diag_indices[offset] = diag.first;
        std::memcpy(
            payload.diag_data + static_cast<std::size_t>(offset) * spec.slots,
            diag.second.data(),
            sizeof(float) * static_cast<std::size_t>(spec.slots));
        ++offset;
      }
    }
    out.payloads[i] = payload;
  }
  return out;
}

OrionDiagPayloadBatch BuildDenseIndexBatch(const DenseSpec &spec, const float *weight, int weight_len) {
  ValidateSpec(spec, weight_len);
  IndexAccumulator acc = MakeIndexAccumulator(spec);
  FillDenseConv2D(spec, weight, acc);

  std::vector<std::pair<int, int>> block_keys;
  for (int row = 0; row < acc.num_block_rows; ++row) {
    for (int col = 0; col < acc.num_block_cols; ++col) {
      block_keys.emplace_back(row, col);
    }
  }

  OrionDiagPayloadBatch out{nullptr, 0, acc.output_rotations, "cpp_dense_conv2d:index_only", nullptr};
  if (block_keys.empty()) {
    return out;
  }
  out.payloads = AllocArray<OrionDiagPayload>(block_keys.size());
  out.len = static_cast<unsigned long>(block_keys.size());
  for (std::size_t i = 0; i < block_keys.size(); ++i) {
    const auto &block_key = block_keys[i];
    const auto block_it = acc.indices.find(block_key);
    const bool empty = block_it == acc.indices.end() || block_it->second.empty();
    std::vector<int> sorted_indices;
    if (!empty) {
      sorted_indices = block_it->second;
      std::sort(sorted_indices.begin(), sorted_indices.end());
    }
    const int diag_count = empty ? 1 : static_cast<int>(sorted_indices.size());
    OrionDiagPayload payload{};
    payload.row = block_key.first;
    payload.col = block_key.second;
    payload.level = 0;
    payload.task_id = nullptr;
    payload.diag_indices = AllocArray<int>(static_cast<std::size_t>(diag_count));
    payload.diag_indices_len = static_cast<unsigned long>(diag_count);
    payload.diag_data = nullptr;
    payload.diag_data_len = 0;
    if (empty) {
      payload.diag_indices[0] = 0;
    } else {
      int offset = 0;
      for (const int diag : sorted_indices) {
        payload.diag_indices[offset] = int(diag);
        ++offset;
      }
    }
    out.payloads[i] = payload;
  }
  return out;
}

OrionDiagPayloadBatch ErrorBatch(const std::string &message) {
  g_last_error = message;
  OrionDiagPayloadBatch out{nullptr, 0, 0, kBuilderKind, g_last_error.c_str()};
  return out;
}

OrionDiagPayloadBatch BuildProviderNativeSourcePayload(
    const OrionProviderNativeSourceSpec &spec,
    const float *weight,
    int weight_len) {
  const int source_start = spec.source_group * spec.source_tile;
  const int source_end = std::min(spec.c_in, source_start + spec.source_tile);
  const int target_start = spec.target_group * spec.target_tile;
  const int target_end = std::min(spec.c_out, target_start + spec.target_tile);
  const int source_count = source_end - source_start;
  const int target_count = target_end - target_start;
  const int64_t expected_weight = static_cast<int64_t>(spec.c_out) * spec.c_in * spec.kernel * spec.kernel;
  if (expected_weight != weight_len) {
    throw std::invalid_argument("provider weight length does not match spec");
  }
  if (source_count <= 0 || target_count <= 0) {
    return OrionDiagPayloadBatch{nullptr, 0, 0, "cpp_provider_native_halo_conv2d:native_source", nullptr};
  }

  std::map<int, std::vector<float>> diagonals;
  for (int kh = 0; kh < spec.kernel; ++kh) {
    for (int kw = 0; kw < spec.kernel; ++kw) {
      for (int64_t out_h = spec.stripe_target_h_start; out_h < spec.stripe_target_h_end; ++out_h) {
        const int64_t op_out_h = spec.compact_output
            ? MaterializedOutputSourceH(out_h, spec.h_out, spec.output_top_beta, spec.output_bottom_beta)
            : out_h;
        const int64_t in_h = op_out_h * spec.stride - spec.pad + static_cast<int64_t>(kh) * spec.dilation;
        const int64_t source_local_h = in_h - spec.stripe_source_h_start;
        const int64_t target_local_h = out_h - spec.stripe_target_h_start;
        if (in_h < spec.input_h_min || in_h >= spec.input_h_max ||
            source_local_h < 0 || source_local_h >= spec.stripe_source_h ||
            target_local_h < 0 || target_local_h >= spec.stripe_target_h) {
          continue;
        }
        for (int out_w = 0; out_w < spec.w_out; ++out_w) {
          const int64_t in_w = static_cast<int64_t>(out_w) * spec.stride - spec.pad + static_cast<int64_t>(kw) * spec.dilation;
          if (in_w < 0 || in_w >= spec.w_in) {
            continue;
          }
          for (int local_source = 0; local_source < source_count; ++local_source) {
            const int64_t source_slot = GapChannelPosition(
                local_source,
                source_local_h,
                in_w,
                spec.stripe_source_h,
                spec.w_in,
                spec.gap_in);
            for (int local_target = 0; local_target < target_count; ++local_target) {
              const int target_channel = spec.compact_output ? target_start + local_target : local_target;
              int64_t target_index = 0;
              bool target_ok = true;
              if (spec.compact_output) {
                target_ok = PhysicalOutputHValid(spec, out_h);
                target_index = GapChannelPosition(
                    target_channel,
                    PhysicalOutputHPosition(spec, out_h),
                    out_w,
                    PhysicalOutputH(spec),
                    spec.w_out,
                    spec.gap_out);
                target_ok = target_ok && (target_index / spec.slots == spec.compact_target_block);
                target_index %= spec.slots;
              } else {
                target_index = GapChannelPosition(
                    target_channel,
                    target_local_h,
                    out_w,
                    spec.stripe_target_h,
                    spec.w_out,
                    spec.gap_out);
              }
              if (!target_ok) {
                continue;
              }
              const int64_t weight_index =
                  (((static_cast<int64_t>(target_start + local_target) * spec.c_in + (source_start + local_source)) * spec.kernel + kh) * spec.kernel + kw);
              const float coeff = weight[weight_index];
              if (coeff == 0.0f) {
                continue;
              }
              int64_t diag = (source_slot - target_index) % spec.slots;
              if (diag < 0) {
                diag += spec.slots;
              }
              std::vector<float> &diag_values = diagonals[static_cast<int>(diag)];
              if (diag_values.empty()) {
                diag_values.assign(static_cast<std::size_t>(spec.slots), 0.0f);
              }
              diag_values[static_cast<std::size_t>(target_index)] += coeff;
            }
          }
        }
      }
    }
  }

  std::vector<int> diag_indices;
  std::vector<float> diag_data;
  for (const auto &item : diagonals) {
    const std::vector<float> &values = item.second;
    const bool nonzero = std::any_of(values.begin(), values.end(), [](float value) {
      return value != 0.0f;
    });
    if (!nonzero) {
      continue;
    }
    diag_indices.push_back(item.first);
    diag_data.insert(diag_data.end(), values.begin(), values.end());
  }
  if (diag_indices.empty()) {
    return OrionDiagPayloadBatch{nullptr, 0, 0, "cpp_provider_native_halo_conv2d:native_source", nullptr};
  }

  OrionDiagPayloadBatch out{nullptr, 0, 0, "cpp_provider_native_halo_conv2d:native_source", nullptr};
  out.payloads = AllocArray<OrionDiagPayload>(1);
  out.len = 1;
  OrionDiagPayload payload{};
  payload.row = spec.target_index;
  payload.col = spec.source_index;
  payload.level = 0;
  payload.task_id = nullptr;
  payload.diag_indices = AllocArray<int>(diag_indices.size());
  payload.diag_indices_len = static_cast<unsigned long>(diag_indices.size());
  payload.diag_data = AllocArray<float>(diag_data.size());
  payload.diag_data_len = static_cast<unsigned long>(diag_data.size());
  std::memcpy(payload.diag_indices, diag_indices.data(), sizeof(int) * diag_indices.size());
  std::memcpy(payload.diag_data, diag_data.data(), sizeof(float) * diag_data.size());
  out.payloads[0] = payload;
  return out;
}

OrionDiagPayloadBatch BuildProviderCompactSourcePayload(
    const OrionProviderCompactSourceSpec &spec,
    const float *weight,
    int weight_len) {
  const int target_start = spec.target_group * spec.target_tile;
  const int target_end = std::min(spec.c_out, target_start + spec.target_tile);
  const int target_count = target_end - target_start;
  const int64_t expected_weight = static_cast<int64_t>(spec.c_out) * spec.c_in * spec.kernel * spec.kernel;
  if (expected_weight != weight_len) {
    throw std::invalid_argument("provider compact-source weight length does not match spec");
  }
  if (target_count <= 0) {
    return OrionDiagPayloadBatch{nullptr, 0, 0, "cpp_provider_native_halo_conv2d:compact_source", nullptr};
  }

  std::map<int64_t, float> rows;
  const int source_height = spec.h_in + std::max(0, spec.source_top_beta) + std::max(0, spec.source_bottom_beta);
  for (int kh = 0; kh < spec.kernel; ++kh) {
    for (int kw = 0; kw < spec.kernel; ++kw) {
      for (int64_t out_h = spec.stripe_target_h_start; out_h < spec.stripe_target_h_end; ++out_h) {
        const int64_t op_out_h = spec.compact_output
            ? MaterializedOutputSourceH(out_h, spec.h_out, spec.output_top_beta, spec.output_bottom_beta)
            : out_h;
        const int64_t in_h = op_out_h * spec.stride - spec.pad + static_cast<int64_t>(kh) * spec.dilation;
        if (in_h < 0 || in_h >= spec.h_in) {
          continue;
        }
        const int64_t source_h = static_cast<int64_t>(std::max(0, spec.source_top_beta)) + in_h;
        const int64_t target_local_h = out_h - spec.stripe_target_h_start;
        if (!spec.compact_output && (target_local_h < 0 || target_local_h >= spec.stripe_target_h)) {
          continue;
        }
        for (int out_w = 0; out_w < spec.w_out; ++out_w) {
          const int64_t in_w = static_cast<int64_t>(out_w) * spec.stride - spec.pad + static_cast<int64_t>(kw) * spec.dilation;
          if (in_w < 0 || in_w >= spec.w_in) {
            continue;
          }
          for (int source_channel = 0; source_channel < spec.c_in; ++source_channel) {
            const int64_t source_index = GapChannelPosition(
                source_channel,
                source_h,
                in_w,
                source_height,
                spec.w_in,
                spec.source_gap);
            if (source_index / spec.slots != spec.source_block) {
              continue;
            }
            const int64_t source_slot = source_index % spec.slots;
            for (int local_target = 0; local_target < target_count; ++local_target) {
              int64_t target_slot = 0;
              bool target_ok = true;
              if (spec.compact_output) {
                const int target_channel = target_start + local_target;
                const int output_top = std::max(0, spec.output_physical_top_beta);
                const int output_bottom = std::max(0, spec.output_physical_bottom_beta);
                target_ok = out_h >= -output_top && out_h < spec.h_out + output_bottom;
                const int compact_output_h =
                    spec.h_out + output_top + output_bottom;
                const int64_t target_index = GapChannelPosition(
                    target_channel,
                    out_h + output_top,
                    out_w,
                    compact_output_h,
                    spec.w_out,
                    spec.gap_out);
                target_ok = target_ok && (target_index / spec.slots == spec.compact_target_block);
                target_slot = target_index % spec.slots;
              } else {
                target_slot = GapChannelPosition(
                    local_target,
                    target_local_h,
                    out_w,
                    spec.stripe_target_h,
                    spec.w_out,
                    spec.gap_out);
              }
              if (!target_ok) {
                continue;
              }
              const int64_t weight_index =
                  (((static_cast<int64_t>(target_start + local_target) * spec.c_in + source_channel) * spec.kernel + kh) * spec.kernel + kw);
              const float coeff = weight[weight_index];
              if (coeff == 0.0f) {
                continue;
              }
              int64_t diag = (source_slot - target_slot) % spec.slots;
              if (diag < 0) {
                diag += spec.slots;
              }
              const int64_t key = diag * static_cast<int64_t>(spec.slots) + target_slot;
              rows[key] += coeff;
            }
          }
        }
      }
    }
  }

  std::vector<std::pair<int64_t, float>> kept;
  kept.reserve(rows.size());
  for (const auto &item : rows) {
    if (item.second != 0.0f) {
      kept.push_back(item);
    }
  }
  if (kept.empty()) {
    return OrionDiagPayloadBatch{nullptr, 0, 0, "cpp_provider_native_halo_conv2d:compact_source", nullptr};
  }

  std::vector<int> diag_indices;
  std::vector<float> diag_data;
  std::vector<float> current(static_cast<std::size_t>(spec.slots), 0.0f);
  int current_diag = -1;
  for (const auto &item : kept) {
    const int diag = static_cast<int>(item.first / spec.slots);
    const int slot = static_cast<int>(item.first % spec.slots);
    if (diag != current_diag) {
      if (current_diag >= 0) {
        diag_data.insert(diag_data.end(), current.begin(), current.end());
        std::fill(current.begin(), current.end(), 0.0f);
      }
      diag_indices.push_back(diag);
      current_diag = diag;
    }
    current[static_cast<std::size_t>(slot)] += item.second;
  }
  diag_data.insert(diag_data.end(), current.begin(), current.end());

  OrionDiagPayloadBatch out{nullptr, 0, 0, "cpp_provider_native_halo_conv2d:compact_source", nullptr};
  out.payloads = AllocArray<OrionDiagPayload>(1);
  out.len = 1;
  OrionDiagPayload payload{};
  payload.row = spec.target_index;
  payload.col = spec.source_block;
  payload.level = 0;
  payload.task_id = nullptr;
  payload.diag_indices = AllocArray<int>(diag_indices.size());
  payload.diag_indices_len = static_cast<unsigned long>(diag_indices.size());
  payload.diag_data = AllocArray<float>(diag_data.size());
  payload.diag_data_len = static_cast<unsigned long>(diag_data.size());
  std::memcpy(payload.diag_indices, diag_indices.data(), sizeof(int) * diag_indices.size());
  std::memcpy(payload.diag_data, diag_data.data(), sizeof(float) * diag_data.size());
  out.payloads[0] = payload;
  return out;
}

}  // namespace

extern "C" {

const char *OrionDiagBuilderVersion() { return "dense_conv2d_v1"; }

const char *OrionDiagBuilderLastError() { return g_last_error.c_str(); }

OrionDiagPayloadBatch OrionBuildDenseConv2D(
    int slots,
    const char *embed_method,
    int is_last_layer,
    int allow_hybrid,
    const int *input_shape,
    const int *output_shape,
    const int *fhe_input_shape,
    const int *fhe_output_shape,
    int input_gap,
    int output_gap,
    int input_row_offset,
    int output_row_offset,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int output_top_beta,
    int output_bottom_beta,
    int fuse_output_relayout,
    const float *weight,
    int weight_len,
    const int *block_rows,
    const int *block_cols,
    int block_count) {
  try {
    g_last_error.clear();
    if (input_shape == nullptr || output_shape == nullptr || fhe_input_shape == nullptr || fhe_output_shape == nullptr || weight == nullptr) {
      return ErrorBatch("null dense Conv2d argument");
    }
    DenseSpec spec;
    spec.slots = int(slots);
    spec.embed_method = embed_method == nullptr ? "" : std::string(embed_method);
    spec.is_last_layer = bool(is_last_layer);
    spec.allow_hybrid = bool(allow_hybrid);
    for (int i = 0; i < 4; ++i) {
      spec.input_shape[i] = input_shape[i];
      spec.output_shape[i] = output_shape[i];
      spec.fhe_input_shape[i] = fhe_input_shape[i];
      spec.fhe_output_shape[i] = fhe_output_shape[i];
    }
    spec.input_gap = input_gap;
    spec.output_gap = output_gap;
    spec.input_row_offset = std::max(0, input_row_offset);
    spec.output_row_offset = std::max(0, output_row_offset);
    spec.kernel_h = kernel_h;
    spec.kernel_w = kernel_w;
    spec.stride_h = stride_h;
    spec.stride_w = stride_w;
    spec.pad_h = pad_h;
    spec.pad_w = pad_w;
    spec.dilation_h = dilation_h;
    spec.dilation_w = dilation_w;
    spec.output_top_beta = std::max(0, output_top_beta);
    spec.output_bottom_beta = std::max(0, output_bottom_beta);
    spec.fuse_output_relayout = bool(fuse_output_relayout);
    return BuildDensePayloadBatch(spec, weight, weight_len, RequestedBlocks(block_rows, block_cols, block_count));
  } catch (const std::exception &exc) {
    return ErrorBatch(exc.what());
  } catch (...) {
    return ErrorBatch("unknown C++ dense Conv2d builder error");
  }
}

OrionDiagPayloadBatch OrionBuildDenseConv2DIndexOnly(
    int slots,
    const char *embed_method,
    int is_last_layer,
    int allow_hybrid,
    const int *input_shape,
    const int *output_shape,
    const int *fhe_input_shape,
    const int *fhe_output_shape,
    int input_gap,
    int output_gap,
    int input_row_offset,
    int output_row_offset,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int output_top_beta,
    int output_bottom_beta,
    int fuse_output_relayout,
    const float *weight,
    int weight_len) {
  try {
    g_last_error.clear();
    if (input_shape == nullptr || output_shape == nullptr || fhe_input_shape == nullptr || fhe_output_shape == nullptr || weight == nullptr) {
      return ErrorBatch("null dense Conv2d index argument");
    }
    DenseSpec spec;
    spec.slots = int(slots);
    spec.embed_method = embed_method == nullptr ? "" : std::string(embed_method);
    spec.is_last_layer = bool(is_last_layer);
    spec.allow_hybrid = bool(allow_hybrid);
    for (int i = 0; i < 4; ++i) {
      spec.input_shape[i] = input_shape[i];
      spec.output_shape[i] = output_shape[i];
      spec.fhe_input_shape[i] = fhe_input_shape[i];
      spec.fhe_output_shape[i] = fhe_output_shape[i];
    }
    spec.input_gap = input_gap;
    spec.output_gap = output_gap;
    spec.input_row_offset = std::max(0, input_row_offset);
    spec.output_row_offset = std::max(0, output_row_offset);
    spec.kernel_h = kernel_h;
    spec.kernel_w = kernel_w;
    spec.stride_h = stride_h;
    spec.stride_w = stride_w;
    spec.pad_h = pad_h;
    spec.pad_w = pad_w;
    spec.dilation_h = dilation_h;
    spec.dilation_w = dilation_w;
    spec.output_top_beta = std::max(0, output_top_beta);
    spec.output_bottom_beta = std::max(0, output_bottom_beta);
    spec.fuse_output_relayout = bool(fuse_output_relayout);
    return BuildDenseIndexBatch(spec, weight, weight_len);
  } catch (const std::exception &exc) {
    return ErrorBatch(exc.what());
  } catch (...) {
    return ErrorBatch("unknown C++ dense Conv2d index builder error");
  }
}

OrionDiagPayloadBatch OrionBuildProviderNativeSourceConv2D(
    OrionProviderNativeSourceSpec spec,
    const float *weight,
    int weight_len) {
  try {
    g_last_error.clear();
    if (weight == nullptr) {
      return ErrorBatch("null provider weight");
    }
    return BuildProviderNativeSourcePayload(spec, weight, weight_len);
  } catch (const std::exception &exc) {
    return ErrorBatch(exc.what());
  } catch (...) {
    return ErrorBatch("unknown C++ provider native-source builder error");
  }
}

OrionDiagPayloadBatch OrionBuildProviderCompactSourceConv2D(
    OrionProviderCompactSourceSpec spec,
    const float *weight,
    int weight_len) {
  try {
    g_last_error.clear();
    if (weight == nullptr) {
      return ErrorBatch("null provider compact-source weight");
    }
    return BuildProviderCompactSourcePayload(spec, weight, weight_len);
  } catch (const std::exception &exc) {
    return ErrorBatch(exc.what());
  } catch (...) {
    return ErrorBatch("unknown C++ provider compact-source builder error");
  }
}

void OrionFreeDiagPayloadBatch(OrionDiagPayloadBatch batch) {
  if (batch.payloads == nullptr) {
    return;
  }
  for (unsigned long i = 0; i < batch.len; ++i) {
    std::free(batch.payloads[i].diag_indices);
    std::free(batch.payloads[i].diag_data);
    std::free(const_cast<char *>(batch.payloads[i].task_id));
  }
  std::free(batch.payloads);
}

}  // extern "C"
