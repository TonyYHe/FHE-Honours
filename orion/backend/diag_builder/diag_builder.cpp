#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
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
  int stripe_source_owner_h_start;
  int stripe_source_owner_h_end;
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

struct OrionProviderStripeSpec {
  int stripe_index;
  int target_h_start;
  int target_h_end;
  int target_h;
  int target_tile;
  int target_group_count;
};

struct OrionProviderCompactSourceConcatIndexSpec {
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
  int source_ct_count;
  int target_ct_count;
  int fuse_output_relayout;
};

}

namespace {

constexpr const char *kDenseConv2DBuilderKind = "cpp_dense_conv2d";
constexpr const char *kDenseConvTranspose2DBuilderKind = "cpp_dense_conv_transpose2d";
constexpr const char *kProviderConcatIndexBuilderKind = "cpp_provider_native_halo_conv2d:compact_source_concat_index_only";

thread_local std::string g_last_error;

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

int64_t FloorDiv(int64_t value, int64_t divisor) {
  if (divisor <= 0) {
    throw std::invalid_argument("invalid divisor");
  }
  int64_t quotient = value / divisor;
  const int64_t remainder = value % divisor;
  if (remainder != 0 && ((remainder < 0) != (divisor < 0))) {
    --quotient;
  }
  return quotient;
}

int64_t PositiveMod(int64_t value, int64_t divisor) {
  if (divisor <= 0) {
    throw std::invalid_argument("invalid divisor");
  }
  int64_t result = value % divisor;
  if (result < 0) {
    result += divisor;
  }
  return result;
}

struct PackedCoords {
  int channel = 0;
  int64_t row = 0;
  int64_t col = 0;
};

bool UnpackPackedFlatIndex(
    int64_t index,
    int gap,
    int height,
    int width,
    int row_offset,
    int channel_limit,
    int64_t row_min,
    int64_t row_max,
    int64_t col_min,
    int64_t col_max,
    PackedCoords &coords) {
  if (index < 0 || gap <= 0 || height <= 0 || width <= 0 || channel_limit <= 0) {
    return false;
  }
  const int64_t packed_col = index % width;
  const int64_t packed_hw = index / width;
  const int64_t packed_row = packed_hw % height;
  const int64_t packed_channel = packed_hw / height;
  const int64_t row_phase = packed_row - row_offset;
  const int64_t phase_h = PositiveMod(row_phase, gap);
  const int64_t phase_w = packed_col % gap;
  const int64_t channel = packed_channel * static_cast<int64_t>(gap) * gap + phase_h * gap + phase_w;
  const int64_t row = FloorDiv(row_phase, gap);
  const int64_t col = packed_col / gap;
  if (channel < 0 || channel >= channel_limit || row < row_min || row >= row_max || col < col_min || col >= col_max) {
    return false;
  }
  coords.channel = static_cast<int>(channel);
  coords.row = row;
  coords.col = col;
  return true;
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

int64_t ChannelBaseOffsetChwGap(int channel, int height, int width, int gap) {
  const int g = std::max(1, gap);
  const int phases = g * g;
  const int packed_w = width * g;
  const int group_block = height * g * packed_w;
  const int group = channel / phases;
  const int phase = channel % phases;
  const int phase_h = phase / g;
  const int phase_w = phase % g;
  return static_cast<int64_t>(group) * group_block + static_cast<int64_t>(phase_h) * packed_w + phase_w;
}

void AddDiagMask(
    std::map<std::pair<int, int>, std::vector<unsigned char>> &masks,
    int source_block,
    int target_block,
    int diag,
    int slots) {
  if (source_block < 0 || target_block < 0 || slots <= 0) {
    return;
  }
  auto &mask = masks[std::make_pair(source_block, target_block)];
  if (mask.empty()) {
    mask.assign(static_cast<std::size_t>(slots), 0);
  }
  mask[static_cast<std::size_t>(diag)] = 1;
}

struct SpatialEvent {
  int64_t source_spatial = 0;
  int64_t target_spatial = 0;
};

struct ConcatIndexWorkItem {
  int stripe_index = 0;
  int target_group = 0;
  int target_start = 0;
  int target_end = 0;
  int kh = 0;
  int kw = 0;
};

int RequestedDiagBuilderWorkers() {
  const char *raw = std::getenv("ORION_CPP_DIAG_BUILDER_PROVIDER_WORKERS");
  if (raw == nullptr || raw[0] == '\0') {
    raw = std::getenv("ORION_PROVIDER_DIAG_BUILD_WORKERS");
  }
  if (raw == nullptr || raw[0] == '\0') {
    const unsigned int hw = std::thread::hardware_concurrency();
    return std::max(1, static_cast<int>(hw == 0 ? 1 : hw));
  }
  try {
    return std::max(1, std::stoi(std::string(raw)));
  } catch (...) {
    return 1;
  }
}

void MergeDiagMasks(
    std::map<std::pair<int, int>, std::vector<unsigned char>> &dst,
    const std::map<std::pair<int, int>, std::vector<unsigned char>> &src,
    int slots) {
  for (const auto &item : src) {
    auto &mask = dst[item.first];
    if (mask.empty()) {
      mask.assign(static_cast<std::size_t>(slots), 0);
    }
    const std::vector<unsigned char> &other = item.second;
    const std::size_t count = std::min(mask.size(), other.size());
    for (std::size_t i = 0; i < count; ++i) {
      mask[i] = static_cast<unsigned char>(mask[i] | other[i]);
    }
  }
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
  struct Block {
    std::pair<int, int> key = {-1, -1};
    std::unordered_map<int, std::vector<float>> diagonals;
  };

  int64_t matrix_height = 0;
  int64_t matrix_width = 0;
  int slots = 0;
  int num_block_rows = 0;
  int num_block_cols = 0;
  int block_height = 0;
  int output_rotations = 0;
  bool restrict_blocks = false;
  std::vector<Block> blocks;
  std::unordered_map<int64_t, int> block_lookup;

  static int64_t BlockLookupKey(int row, int col) {
    return (static_cast<int64_t>(row) << 32) ^ static_cast<unsigned int>(col);
  }

  void InitBlocks(const std::vector<std::pair<int, int>> &block_keys, bool restricted) {
    restrict_blocks = bool(restricted);
    blocks.clear();
    blocks.reserve(block_keys.size());
    block_lookup.clear();
    block_lookup.reserve(block_keys.size());
    for (const auto &key : block_keys) {
      const int index = static_cast<int>(blocks.size());
      Block block;
      block.key = key;
      blocks.push_back(std::move(block));
      block_lookup.emplace(BlockLookupKey(key.first, key.second), index);
    }
  }

  Block *FindBlock(int row, int col) {
    if (row < 0 || row >= num_block_rows || col < 0 || col >= num_block_cols) {
      return nullptr;
    }
    if (!restrict_blocks) {
      return &blocks[static_cast<std::size_t>(row * num_block_cols + col)];
    }
    const auto it = block_lookup.find(BlockLookupKey(row, col));
    if (it == block_lookup.end()) {
      return nullptr;
    }
    return &blocks[static_cast<std::size_t>(it->second)];
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

    Block *block = FindBlock(block_row, block_col);
    if (block == nullptr) {
      return;
    }
    auto it = block->diagonals.find(diag_idx);
    if (it == block->diagonals.end()) {
      it = block->diagonals.emplace(diag_idx, std::vector<float>(static_cast<std::size_t>(slots), 0.0f)).first;
    }
    it->second[static_cast<std::size_t>(position)] += value;
  }
};

struct RequestedBlockFilter {
  std::map<int, std::vector<int>> cols_by_row;

  static bool ContainsCol(const std::vector<int> &cols, int col) {
    return std::binary_search(cols.begin(), cols.end(), col);
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

std::vector<std::pair<int, int>> ValidBlockKeys(
    int num_block_rows,
    int num_block_cols,
    const std::vector<std::pair<int, int>> &blocks);

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
  acc.InitBlocks(ValidBlockKeys(acc.num_block_rows, acc.num_block_cols, blocks), !blocks.empty());
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

std::vector<std::pair<int, int>> ValidBlockKeys(
    int num_block_rows,
    int num_block_cols,
    const std::vector<std::pair<int, int>> &blocks) {
  std::vector<std::pair<int, int>> block_keys;
  if (blocks.empty()) {
    for (int row = 0; row < num_block_rows; ++row) {
      for (int col = 0; col < num_block_cols; ++col) {
        block_keys.emplace_back(row, col);
      }
    }
  } else {
    for (const auto &block : blocks) {
      if (block.first >= 0 && block.first < num_block_rows && block.second >= 0 && block.second < num_block_cols) {
        block_keys.push_back(block);
      }
    }
  }
  std::sort(block_keys.begin(), block_keys.end());
  block_keys.erase(std::unique(block_keys.begin(), block_keys.end()), block_keys.end());
  return block_keys;
}

RequestedBlockFilter MakeRequestedBlockFilter(
    int num_block_rows,
    int num_block_cols,
    const std::vector<std::pair<int, int>> &blocks) {
  RequestedBlockFilter filter;
  for (const auto &block : ValidBlockKeys(num_block_rows, num_block_cols, blocks)) {
    filter.cols_by_row[block.first].push_back(block.second);
  }
  for (auto &entry : filter.cols_by_row) {
    std::sort(entry.second.begin(), entry.second.end());
    entry.second.erase(std::unique(entry.second.begin(), entry.second.end()), entry.second.end());
  }
  return filter;
}

int InputBlockColFor(const Accumulator &acc, int64_t col) {
  if (col < 0 || col >= acc.matrix_width) {
    return -1;
  }
  return static_cast<int>(col / acc.slots);
}

std::pair<int64_t, int64_t> OutputBlockRowRange(const Accumulator &acc, int block_row) {
  if (block_row < 0 || block_row >= acc.num_block_rows) {
    return std::make_pair<int64_t, int64_t>(0, 0);
  }
  if (acc.block_height == acc.slots) {
    const int64_t start = static_cast<int64_t>(block_row) * acc.slots;
    const int64_t end = std::min(acc.matrix_height, start + acc.slots);
    return std::make_pair(start, end);
  }
  if (block_row != 0) {
    return std::make_pair<int64_t, int64_t>(0, 0);
  }
  return std::make_pair(static_cast<int64_t>(0), acc.matrix_height);
}

bool RequestedColsContainEntry(
    const Accumulator &acc,
    const std::vector<int> &requested_cols,
    int64_t col) {
  const int block_col = InputBlockColFor(acc, col);
  return block_col >= 0 && RequestedBlockFilter::ContainsCol(requested_cols, block_col);
}

void ValidateDenseConv2DSpec(const DenseSpec &spec, int weight_len) {
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

void ValidateDenseConvTranspose2DSpec(const DenseSpec &spec, int weight_len) {
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
      static_cast<int64_t>(spec.input_shape[1]) * static_cast<int64_t>(spec.output_shape[1]) *
      spec.kernel_h * spec.kernel_w;
  if (expected_weight != weight_len) {
    throw std::invalid_argument("weight length does not match dense ConvTranspose2d spec");
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

void FillDenseConv2DRequestedBlocks(
    const DenseSpec &spec,
    const float *weight,
    Accumulator &acc,
    const RequestedBlockFilter &filter) {
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

  for (const auto &row_entry : filter.cols_by_row) {
    const int block_row = row_entry.first;
    const std::vector<int> &requested_cols = row_entry.second;
    const auto row_range = OutputBlockRowRange(acc, block_row);
    if (row_range.first >= row_range.second || requested_cols.empty()) {
      continue;
    }
    for (int64_t row = row_range.first; row < row_range.second; ++row) {
      const int batch = static_cast<int>(row / output_block_size);
      if (batch < 0 || batch >= n_batch) {
        continue;
      }
      const int64_t local_row = row - static_cast<int64_t>(batch) * output_block_size;
      PackedCoords output{};
      if (!UnpackPackedFlatIndex(
              local_row,
              spec.output_gap,
              on_ho,
              on_wo,
              spec.output_row_offset,
              co,
              -spec.output_top_beta,
              static_cast<int64_t>(ho) + spec.output_bottom_beta,
              0,
              wo,
              output)) {
        continue;
      }
      int64_t op_oh = output.row;
      if (spec.fuse_output_relayout) {
        if (op_oh < 0) {
          op_oh += spec.output_top_beta;
        } else if (op_oh >= ho) {
          op_oh -= spec.output_bottom_beta;
        }
        op_oh = std::min<int64_t>(std::max<int64_t>(op_oh, 0), std::max(0, ho - 1));
      }
      for (int ic = 0; ic < ci; ++ic) {
        for (int kh = 0; kh < spec.kernel_h; ++kh) {
          const int64_t ih = op_oh * spec.stride_h - spec.pad_h + static_cast<int64_t>(kh) * spec.dilation_h;
          if (ih < 0 || ih >= hi) {
            continue;
          }
          for (int kw = 0; kw < spec.kernel_w; ++kw) {
            const int64_t iw_value = output.col * spec.stride_w - spec.pad_w + static_cast<int64_t>(kw) * spec.dilation_w;
            if (iw_value < 0 || iw_value >= wi) {
              continue;
            }
            const int64_t weight_index =
                (((static_cast<int64_t>(output.channel) * ci + ic) * spec.kernel_h + kh) * spec.kernel_w + kw);
            const float coeff = weight[weight_index];
            if (coeff == 0.0f) {
              continue;
            }
            const int64_t local_col = PackedFlatIndex(
                ic,
                ih,
                iw_value,
                spec.input_gap,
                on_hi,
                on_wi,
                spec.input_row_offset);
            const int64_t col = local_col + static_cast<int64_t>(batch) * input_block_size;
            if (!RequestedColsContainEntry(acc, requested_cols, col)) {
              continue;
            }
            acc.AddEntry(row, col, coeff);
          }
        }
      }
    }
  }
}

template <typename TAccumulator>
void FillDenseConvTranspose2D(const DenseSpec &spec, const float *weight, TAccumulator &acc) {
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

  for (int ic = 0; ic < ci; ++ic) {
    for (int oc = 0; oc < co; ++oc) {
      for (int kh = 0; kh < spec.kernel_h; ++kh) {
        for (int kw = 0; kw < spec.kernel_w; ++kw) {
          const int64_t weight_index =
              (((static_cast<int64_t>(ic) * co + oc) * spec.kernel_h + kh) * spec.kernel_w + kw);
          const float coeff = weight[weight_index];
          if (coeff == 0.0f) {
            continue;
          }
          for (int ih = 0; ih < hi; ++ih) {
            const int64_t oh = static_cast<int64_t>(ih) * spec.stride_h - spec.pad_h + static_cast<int64_t>(kh) * spec.dilation_h;
            if (oh < 0 || oh >= ho) {
              continue;
            }
            for (int iw = 0; iw < wi; ++iw) {
              const int64_t ow = static_cast<int64_t>(iw) * spec.stride_w - spec.pad_w + static_cast<int64_t>(kw) * spec.dilation_w;
              if (ow < 0 || ow >= wo) {
                continue;
              }
              const int64_t local_row = PackedFlatIndex(
                  oc,
                  oh,
                  ow,
                  spec.output_gap,
                  on_ho,
                  on_wo,
                  spec.output_row_offset);
              const int64_t local_col = PackedFlatIndex(
                  ic,
                  ih,
                  iw,
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

void FillDenseConvTranspose2DRequestedBlocks(
    const DenseSpec &spec,
    const float *weight,
    Accumulator &acc,
    const RequestedBlockFilter &filter) {
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

  for (const auto &row_entry : filter.cols_by_row) {
    const int block_row = row_entry.first;
    const std::vector<int> &requested_cols = row_entry.second;
    const auto row_range = OutputBlockRowRange(acc, block_row);
    if (row_range.first >= row_range.second || requested_cols.empty()) {
      continue;
    }
    for (int64_t row = row_range.first; row < row_range.second; ++row) {
      const int batch = static_cast<int>(row / output_block_size);
      if (batch < 0 || batch >= n_batch) {
        continue;
      }
      const int64_t local_row = row - static_cast<int64_t>(batch) * output_block_size;
      PackedCoords output{};
      if (!UnpackPackedFlatIndex(
              local_row,
              spec.output_gap,
              on_ho,
              on_wo,
              spec.output_row_offset,
              co,
              0,
              ho,
              0,
              wo,
              output)) {
        continue;
      }
      for (int ic = 0; ic < ci; ++ic) {
        for (int kh = 0; kh < spec.kernel_h; ++kh) {
          const int64_t numer_h = output.row + spec.pad_h - static_cast<int64_t>(kh) * spec.dilation_h;
          if (numer_h % spec.stride_h != 0) {
            continue;
          }
          const int64_t ih = numer_h / spec.stride_h;
          if (ih < 0 || ih >= hi) {
            continue;
          }
          for (int kw = 0; kw < spec.kernel_w; ++kw) {
            const int64_t numer_w = output.col + spec.pad_w - static_cast<int64_t>(kw) * spec.dilation_w;
            if (numer_w % spec.stride_w != 0) {
              continue;
            }
            const int64_t iw_value = numer_w / spec.stride_w;
            if (iw_value < 0 || iw_value >= wi) {
              continue;
            }
            const int64_t weight_index =
                (((static_cast<int64_t>(ic) * co + output.channel) * spec.kernel_h + kh) * spec.kernel_w + kw);
            const float coeff = weight[weight_index];
            if (coeff == 0.0f) {
              continue;
            }
            const int64_t local_col = PackedFlatIndex(
                ic,
                ih,
                iw_value,
                spec.input_gap,
                on_hi,
                on_wi,
                spec.input_row_offset);
            const int64_t col = local_col + static_cast<int64_t>(batch) * input_block_size;
            if (!RequestedColsContainEntry(acc, requested_cols, col)) {
              continue;
            }
            acc.AddEntry(row, col, coeff);
          }
        }
      }
    }
  }
}

OrionDiagPayloadBatch BuildDensePayloadBatch(const DenseSpec &spec, const float *weight, int weight_len, const std::vector<std::pair<int, int>> &blocks) {
  ValidateDenseConv2DSpec(spec, weight_len);
  Accumulator acc = MakeAccumulator(spec, blocks);
  if (blocks.empty()) {
    FillDenseConv2D(spec, weight, acc);
  } else {
    FillDenseConv2DRequestedBlocks(spec, weight, acc, MakeRequestedBlockFilter(acc.num_block_rows, acc.num_block_cols, blocks));
  }
  std::vector<std::pair<int, int>> block_keys = ValidBlockKeys(acc.num_block_rows, acc.num_block_cols, blocks);

  OrionDiagPayloadBatch out{nullptr, 0, acc.output_rotations, kDenseConv2DBuilderKind, nullptr};
  if (block_keys.empty()) {
    return out;
  }
  out.payloads = AllocArray<OrionDiagPayload>(block_keys.size());
  out.len = static_cast<unsigned long>(block_keys.size());
  for (std::size_t i = 0; i < block_keys.size(); ++i) {
    const auto &block_key = block_keys[i];
    Accumulator::Block *block = acc.FindBlock(block_key.first, block_key.second);
    const bool empty = block == nullptr || block->diagonals.empty();
    std::vector<int> sorted_indices;
    if (!empty) {
      sorted_indices.reserve(block->diagonals.size());
      for (const auto &diag : block->diagonals) {
        sorted_indices.push_back(diag.first);
      }
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
    payload.diag_data = AllocArray<float>(static_cast<std::size_t>(diag_count) * static_cast<std::size_t>(spec.slots));
    payload.diag_data_len = static_cast<unsigned long>(diag_count * spec.slots);
    if (empty) {
      payload.diag_indices[0] = 0;
      std::fill(payload.diag_data, payload.diag_data + spec.slots, 0.0f);
    } else {
      int offset = 0;
      for (const int diag_idx : sorted_indices) {
        const auto diag_it = block->diagonals.find(diag_idx);
        payload.diag_indices[offset] = diag_idx;
        std::memcpy(
            payload.diag_data + static_cast<std::size_t>(offset) * spec.slots,
            diag_it->second.data(),
            sizeof(float) * static_cast<std::size_t>(spec.slots));
        ++offset;
      }
    }
    out.payloads[i] = payload;
  }
  return out;
}

OrionDiagPayloadBatch BuildDenseIndexBatch(const DenseSpec &spec, const float *weight, int weight_len) {
  ValidateDenseConv2DSpec(spec, weight_len);
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

OrionDiagPayloadBatch BuildDenseConvTransposePayloadBatch(const DenseSpec &spec, const float *weight, int weight_len, const std::vector<std::pair<int, int>> &blocks) {
  ValidateDenseConvTranspose2DSpec(spec, weight_len);
  Accumulator acc = MakeAccumulator(spec, blocks);
  if (blocks.empty()) {
    FillDenseConvTranspose2D(spec, weight, acc);
  } else {
    FillDenseConvTranspose2DRequestedBlocks(spec, weight, acc, MakeRequestedBlockFilter(acc.num_block_rows, acc.num_block_cols, blocks));
  }
  std::vector<std::pair<int, int>> block_keys = ValidBlockKeys(acc.num_block_rows, acc.num_block_cols, blocks);

  OrionDiagPayloadBatch out{nullptr, 0, acc.output_rotations, kDenseConvTranspose2DBuilderKind, nullptr};
  if (block_keys.empty()) {
    return out;
  }
  out.payloads = AllocArray<OrionDiagPayload>(block_keys.size());
  out.len = static_cast<unsigned long>(block_keys.size());
  for (std::size_t i = 0; i < block_keys.size(); ++i) {
    const auto &block_key = block_keys[i];
    Accumulator::Block *block = acc.FindBlock(block_key.first, block_key.second);
    const bool empty = block == nullptr || block->diagonals.empty();
    std::vector<int> sorted_indices;
    if (!empty) {
      sorted_indices.reserve(block->diagonals.size());
      for (const auto &diag : block->diagonals) {
        sorted_indices.push_back(diag.first);
      }
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
    payload.diag_data = AllocArray<float>(static_cast<std::size_t>(diag_count) * static_cast<std::size_t>(spec.slots));
    payload.diag_data_len = static_cast<unsigned long>(diag_count * spec.slots);
    if (empty) {
      payload.diag_indices[0] = 0;
      std::fill(payload.diag_data, payload.diag_data + spec.slots, 0.0f);
    } else {
      int offset = 0;
      for (const int diag_idx : sorted_indices) {
        const auto diag_it = block->diagonals.find(diag_idx);
        payload.diag_indices[offset] = diag_idx;
        std::memcpy(
            payload.diag_data + static_cast<std::size_t>(offset) * spec.slots,
            diag_it->second.data(),
            sizeof(float) * static_cast<std::size_t>(spec.slots));
        ++offset;
      }
    }
    out.payloads[i] = payload;
  }
  return out;
}

OrionDiagPayloadBatch BuildDenseConvTransposeIndexBatch(const DenseSpec &spec, const float *weight, int weight_len) {
  ValidateDenseConvTranspose2DSpec(spec, weight_len);
  IndexAccumulator acc = MakeIndexAccumulator(spec);
  FillDenseConvTranspose2D(spec, weight, acc);

  std::vector<std::pair<int, int>> block_keys;
  for (int row = 0; row < acc.num_block_rows; ++row) {
    for (int col = 0; col < acc.num_block_cols; ++col) {
      block_keys.emplace_back(row, col);
    }
  }

  OrionDiagPayloadBatch out{nullptr, 0, acc.output_rotations, "cpp_dense_conv_transpose2d:index_only", nullptr};
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

OrionDiagPayloadBatch ErrorBatch(const std::string &message, const char *builder_kind = kDenseConv2DBuilderKind) {
  g_last_error = message;
  OrionDiagPayloadBatch out{nullptr, 0, 0, builder_kind, g_last_error.c_str()};
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
            in_h < spec.stripe_source_owner_h_start ||
            in_h >= spec.stripe_source_owner_h_end ||
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

OrionDiagPayloadBatch BuildProviderCompactSourceConcatIndexOnly(
    const OrionProviderCompactSourceConcatIndexSpec &spec,
    const OrionProviderStripeSpec *stripes,
    int stripe_count,
    const float *weight,
    int weight_len) {
  const int64_t expected_weight = static_cast<int64_t>(spec.c_out) * spec.c_in * spec.kernel * spec.kernel;
  if (expected_weight != weight_len) {
    throw std::invalid_argument("provider compact-source concat index weight length does not match spec");
  }
  if (stripes == nullptr && stripe_count > 0) {
    throw std::invalid_argument("provider compact-source concat index stripes are required");
  }
  if (spec.source_ct_count <= 0 || spec.target_ct_count <= 0 || spec.slots <= 0) {
    return OrionDiagPayloadBatch{nullptr, 0, 0, kProviderConcatIndexBuilderKind, nullptr};
  }

  const int source_height = spec.h_in + std::max(0, spec.source_top_beta) + std::max(0, spec.source_bottom_beta);
  const int compact_output_h = spec.h_out + std::max(0, spec.output_physical_top_beta) + std::max(0, spec.output_physical_bottom_beta);
  const int source_gap = std::max(1, spec.source_gap);
  const int target_gap = std::max(1, spec.gap_out);
  const int source_packed_w = spec.w_in * source_gap;
  const int target_packed_w = spec.w_out * target_gap;
  std::vector<int64_t> source_channel_bases(static_cast<std::size_t>(spec.c_in), 0);
  for (int channel = 0; channel < spec.c_in; ++channel) {
    source_channel_bases[static_cast<std::size_t>(channel)] =
        ChannelBaseOffsetChwGap(channel, source_height, spec.w_in, source_gap);
  }
  std::vector<int64_t> target_channel_bases(static_cast<std::size_t>(spec.c_out), 0);
  for (int channel = 0; channel < spec.c_out; ++channel) {
    target_channel_bases[static_cast<std::size_t>(channel)] =
        ChannelBaseOffsetChwGap(channel, compact_output_h, spec.w_out, target_gap);
  }

  std::vector<ConcatIndexWorkItem> work_items;
  for (int stripe_i = 0; stripe_i < stripe_count; ++stripe_i) {
    const OrionProviderStripeSpec &stripe = stripes[stripe_i];
    if (stripe.target_h_end <= stripe.target_h_start || stripe.target_tile <= 0 || stripe.target_group_count <= 0) {
      continue;
    }
    for (int target_group = 0; target_group < stripe.target_group_count; ++target_group) {
      const int target_start = target_group * stripe.target_tile;
      const int target_end = std::min(spec.c_out, target_start + stripe.target_tile);
      if (target_end <= target_start) {
        continue;
      }
      for (int kh = 0; kh < spec.kernel; ++kh) {
        for (int kw = 0; kw < spec.kernel; ++kw) {
          work_items.push_back(ConcatIndexWorkItem{stripe_i, target_group, target_start, target_end, kh, kw});
        }
      }
    }
  }

  std::map<std::pair<int, int>, std::vector<unsigned char>> masks;
  if (work_items.empty()) {
    return OrionDiagPayloadBatch{nullptr, 0, 0, kProviderConcatIndexBuilderKind, nullptr};
  }

  const int worker_count = std::max(1, std::min(static_cast<int>(work_items.size()), RequestedDiagBuilderWorkers()));
  auto process_range = [&](int begin, int end) {
    std::map<std::pair<int, int>, std::vector<unsigned char>> local_masks;
    for (int work_index = begin; work_index < end; ++work_index) {
      const ConcatIndexWorkItem &work = work_items[static_cast<std::size_t>(work_index)];
      const OrionProviderStripeSpec &stripe = stripes[work.stripe_index];
      const int target_start = work.target_start;
      const int target_end = work.target_end;
      const int kh = work.kh;
      const int kw = work.kw;
          std::vector<SpatialEvent> events;
          events.reserve(static_cast<std::size_t>(std::max(0, stripe.target_h_end - stripe.target_h_start) * std::max(0, spec.w_out)));
          for (int64_t out_h = stripe.target_h_start; out_h < stripe.target_h_end; ++out_h) {
            const int64_t op_out_h = spec.fuse_output_relayout
                ? MaterializedOutputSourceH(out_h, spec.h_out, spec.output_top_beta, spec.output_bottom_beta)
                : out_h;
            const int64_t in_h = op_out_h * spec.stride - spec.pad + static_cast<int64_t>(kh) * spec.dilation;
            if (in_h < 0 || in_h >= spec.h_in) {
              continue;
            }
            const int64_t source_h = static_cast<int64_t>(std::max(0, spec.source_top_beta)) + in_h;
            const bool target_h_valid = out_h >= -std::max(0, spec.output_physical_top_beta) &&
                out_h < spec.h_out + std::max(0, spec.output_physical_bottom_beta);
            if (!target_h_valid) {
              continue;
            }
            const int64_t target_h = out_h + std::max(0, spec.output_physical_top_beta);
            if (target_h < 0 || target_h >= compact_output_h) {
              continue;
            }
            for (int out_w = 0; out_w < spec.w_out; ++out_w) {
              const int64_t in_w = static_cast<int64_t>(out_w) * spec.stride - spec.pad + static_cast<int64_t>(kw) * spec.dilation;
              if (in_w < 0 || in_w >= spec.w_in) {
                continue;
              }
              events.push_back(SpatialEvent{
                  source_h * static_cast<int64_t>(source_gap) * source_packed_w + in_w * static_cast<int64_t>(source_gap),
                  target_h * static_cast<int64_t>(target_gap) * target_packed_w + static_cast<int64_t>(out_w) * target_gap,
              });
            }
          }
          if (events.empty()) {
            continue;
          }
          for (int source_channel = 0; source_channel < spec.c_in; ++source_channel) {
            const int64_t source_channel_base = source_channel_bases[static_cast<std::size_t>(source_channel)];
            for (int target_channel = target_start; target_channel < target_end; ++target_channel) {
              const int64_t weight_index =
                  (((static_cast<int64_t>(target_channel) * spec.c_in + source_channel) * spec.kernel + kh) * spec.kernel + kw);
              if (weight[weight_index] == 0.0f) {
                continue;
              }
              const int64_t target_channel_base = target_channel_bases[static_cast<std::size_t>(target_channel)];
              for (const SpatialEvent &event : events) {
                const int64_t source_index = source_channel_base + event.source_spatial;
                const int source_block = static_cast<int>(source_index / spec.slots);
                if (source_block < 0 || source_block >= spec.source_ct_count) {
                  continue;
                }
                const int source_slot = static_cast<int>(source_index % spec.slots);
                const int64_t target_index = target_channel_base + event.target_spatial;
                const int target_block = static_cast<int>(target_index / spec.slots);
                if (target_block < 0 || target_block >= spec.target_ct_count) {
                  continue;
                }
                const int target_slot = static_cast<int>(target_index % spec.slots);
                int diag = (source_slot - target_slot) % spec.slots;
                if (diag < 0) {
                  diag += spec.slots;
                }
                AddDiagMask(local_masks, source_block, target_block, diag, spec.slots);
              }
            }
          }
    }
    return local_masks;
  };

  if (worker_count <= 1) {
    masks = process_range(0, static_cast<int>(work_items.size()));
  } else {
    std::vector<std::map<std::pair<int, int>, std::vector<unsigned char>>> partials(static_cast<std::size_t>(worker_count));
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(worker_count));
    for (int worker = 0; worker < worker_count; ++worker) {
      const int begin = static_cast<int>((static_cast<int64_t>(work_items.size()) * worker) / worker_count);
      const int end = static_cast<int>((static_cast<int64_t>(work_items.size()) * (worker + 1)) / worker_count);
      threads.emplace_back([&, worker, begin, end]() {
        partials[static_cast<std::size_t>(worker)] = process_range(begin, end);
      });
    }
    for (std::thread &thread : threads) {
      thread.join();
    }
    for (const auto &partial : partials) {
      MergeDiagMasks(masks, partial, spec.slots);
    }
  }


  std::vector<std::pair<std::pair<int, int>, std::vector<int>>> payload_indices;
  payload_indices.reserve(masks.size());
  for (const auto &item : masks) {
    std::vector<int> indices;
    const std::vector<unsigned char> &mask = item.second;
    for (int diag = 0; diag < static_cast<int>(mask.size()); ++diag) {
      if (mask[static_cast<std::size_t>(diag)] != 0) {
        indices.push_back(diag);
      }
    }
    if (!indices.empty()) {
      payload_indices.emplace_back(item.first, std::move(indices));
    }
  }

  OrionDiagPayloadBatch out{nullptr, 0, 0, kProviderConcatIndexBuilderKind, nullptr};
  if (payload_indices.empty()) {
    return out;
  }
  out.payloads = AllocArray<OrionDiagPayload>(payload_indices.size());
  out.len = static_cast<unsigned long>(payload_indices.size());
  for (std::size_t i = 0; i < payload_indices.size(); ++i) {
    const auto &entry = payload_indices[i];
    const std::vector<int> &indices = entry.second;
    OrionDiagPayload payload{};
    payload.row = entry.first.second;
    payload.col = entry.first.first;
    payload.level = 0;
    payload.task_id = nullptr;
    payload.diag_indices = AllocArray<int>(indices.size());
    payload.diag_indices_len = static_cast<unsigned long>(indices.size());
    payload.diag_data = nullptr;
    payload.diag_data_len = 0;
    std::memcpy(payload.diag_indices, indices.data(), sizeof(int) * indices.size());
    out.payloads[i] = payload;
  }
  return out;
}

}  // namespace

extern "C" {

const char *OrionDiagBuilderVersion() { return "dense_conv2d_conv_transpose2d_v1"; }

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

OrionDiagPayloadBatch OrionBuildDenseConvTranspose2D(
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
      return ErrorBatch("null dense ConvTranspose2d argument", kDenseConvTranspose2DBuilderKind);
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
    return BuildDenseConvTransposePayloadBatch(spec, weight, weight_len, RequestedBlocks(block_rows, block_cols, block_count));
  } catch (const std::exception &exc) {
    return ErrorBatch(exc.what(), kDenseConvTranspose2DBuilderKind);
  } catch (...) {
    return ErrorBatch("unknown C++ dense ConvTranspose2d builder error", kDenseConvTranspose2DBuilderKind);
  }
}

OrionDiagPayloadBatch OrionBuildDenseConvTranspose2DIndexOnly(
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
      return ErrorBatch("null dense ConvTranspose2d index argument", "cpp_dense_conv_transpose2d:index_only");
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
    return BuildDenseConvTransposeIndexBatch(spec, weight, weight_len);
  } catch (const std::exception &exc) {
    return ErrorBatch(exc.what(), "cpp_dense_conv_transpose2d:index_only");
  } catch (...) {
    return ErrorBatch("unknown C++ dense ConvTranspose2d index builder error", "cpp_dense_conv_transpose2d:index_only");
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

OrionDiagPayloadBatch OrionBuildProviderCompactSourceConcatConv2DIndexOnly(
    OrionProviderCompactSourceConcatIndexSpec spec,
    const OrionProviderStripeSpec *stripes,
    int stripe_count,
    const float *weight,
    int weight_len) {
  try {
    g_last_error.clear();
    if (weight == nullptr) {
      return ErrorBatch("null provider compact-source concat index weight", kProviderConcatIndexBuilderKind);
    }
    return BuildProviderCompactSourceConcatIndexOnly(spec, stripes, stripe_count, weight, weight_len);
  } catch (const std::exception &exc) {
    return ErrorBatch(exc.what(), kProviderConcatIndexBuilderKind);
  } catch (...) {
    return ErrorBatch("unknown C++ provider compact-source concat index builder error", kProviderConcatIndexBuilderKind);
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
