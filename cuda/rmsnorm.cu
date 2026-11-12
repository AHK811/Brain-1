#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t error_ = (call);                                                \
    if (error_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "cuda error %s:%d %s\n", __FILE__, __LINE__,         \
                   cudaGetErrorString(error_));                                 \
      std::exit(2);                                                             \
    }                                                                           \
  } while (0)

struct Options {
  std::string x_path, w_path, y_path, out_path, key;
  long rows = 0;
  long cols = 0;
  float eps = 1e-6f;
  long reps = 200;
  long warmup = 20;
  bool fp16 = false;
};

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ void load_vector(const float *pointer, float *values) {
  const float4 loaded = *reinterpret_cast<const float4 *>(pointer);
  values[0] = loaded.x;
  values[1] = loaded.y;
  values[2] = loaded.z;
  values[3] = loaded.w;
}

__device__ __forceinline__ void store_vector(float *pointer, const float *values) {
  *reinterpret_cast<float4 *>(pointer) = make_float4(values[0], values[1], values[2], values[3]);
}

__device__ __forceinline__ void load_vector(const __half *pointer, float *values) {
  const uint4 loaded = *reinterpret_cast<const uint4 *>(pointer);
  const __half *parts = reinterpret_cast<const __half *>(&loaded);
#pragma unroll
  for (int lane = 0; lane < 8; ++lane) values[lane] = __half2float(parts[lane]);
}

__device__ __forceinline__ void store_vector(__half *pointer, const float *values) {
  __half parts[8];
#pragma unroll
  for (int lane = 0; lane < 8; ++lane) parts[lane] = __float2half_rn(values[lane]);
  *reinterpret_cast<uint4 *>(pointer) = *reinterpret_cast<const uint4 *>(parts);
}

__device__ __forceinline__ float to_scalar(float value) { return value; }

__device__ __forceinline__ float to_scalar(const __half &value) { return __half2float(value); }

__device__ __forceinline__ float from_scalar(float value) { return value; }

__device__ __forceinline__ __half from_scalar(const float value) { return __float2half_rn(value); }

template <typename T, int LANES>
__global__ void rmsnorm_kernel(const T *__restrict__ x, const T *__restrict__ w, T *__restrict__ y,
                               int cols, float eps) {
  const int row = blockIdx.x;
  const T *xrow = x + (size_t)row * cols;
  T *yrow = y + (size_t)row * cols;
  const int vectors = cols / LANES;

  float partial = 0.0f;
  for (int index = threadIdx.x; index < vectors; index += blockDim.x) {
    float values[LANES];
    load_vector(xrow + index * LANES, values);
#pragma unroll
    for (int lane = 0; lane < LANES; ++lane) partial += values[lane] * values[lane];
  }

  __shared__ float warp_sums[32];
  __shared__ float row_scale;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  float total = warp_reduce_sum(partial);
  if (lane == 0) warp_sums[warp] = total;
  __syncthreads();
  const int warps = (blockDim.x + 31) >> 5;
  if (warp == 0) {
    float value = (lane < warps) ? warp_sums[lane] : 0.0f;
    value = warp_reduce_sum(value);
    if (lane == 0) row_scale = rsqrtf(value / (float)cols + eps);
  }
  __syncthreads();

  const float scale = row_scale;
  for (int index = threadIdx.x; index < vectors; index += blockDim.x) {
    float values[LANES];
    float weights[LANES];
    load_vector(xrow + index * LANES, values);
    load_vector(w + index * LANES, weights);
#pragma unroll
    for (int inner = 0; inner < LANES; ++inner) values[inner] *= scale * weights[inner];
    store_vector(yrow + index * LANES, values);
  }
}

static void usage(const char *argv0) {
  std::fprintf(stderr,
               "usage: %s --key NAME --x FILE --w FILE --rows N --cols N\n"
               "          [--y FILE] [--out FILE] [--eps F] [--reps N] [--warmup N]\n"
               "          [--dtype f32|f16]\n",
               argv0);
}

static bool parse(int argc, char **argv, Options &o) {
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto next_string = [&](std::string &slot) {
      if (i + 1 >= argc) return false;
      slot = argv[++i];
      return true;
    };
    auto next_long = [&](long &slot) {
      if (i + 1 >= argc) return false;
      slot = std::strtol(argv[++i], nullptr, 10);
      return true;
    };
    if (key == "--key") {
      if (!next_string(o.key)) return false;
    } else if (key == "--x") {
      if (!next_string(o.x_path)) return false;
    } else if (key == "--w") {
      if (!next_string(o.w_path)) return false;
    } else if (key == "--y") {
      if (!next_string(o.y_path)) return false;
    } else if (key == "--out") {
      if (!next_string(o.out_path)) return false;
    } else if (key == "--rows") {
      if (!next_long(o.rows)) return false;
    } else if (key == "--cols") {
      if (!next_long(o.cols)) return false;
    } else if (key == "--reps") {
      if (!next_long(o.reps)) return false;
    } else if (key == "--warmup") {
      if (!next_long(o.warmup)) return false;
    } else if (key == "--eps") {
      std::string value;
      if (!next_string(value)) return false;
      o.eps = std::strtof(value.c_str(), nullptr);
    } else if (key == "--dtype") {
      std::string dtype;
      if (!next_string(dtype)) return false;
      o.fp16 = (dtype == "f16" || dtype == "fp16");
    } else {
      std::fprintf(stderr, "unknown argument %s\n", key.c_str());
      return false;
    }
  }
  if (o.x_path.empty() || o.w_path.empty() || o.rows <= 0 || o.cols <= 0) {
    usage(argv[0]);
    return false;
  }
  return true;
}

template <typename T>
static std::vector<T> read_binary(const std::string &path, size_t expected) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(2);
  }
  std::vector<T> data(expected);
  stream.read(reinterpret_cast<char *>(data.data()), (std::streamsize)(expected * sizeof(T)));
  if ((size_t)stream.gcount() != expected * sizeof(T)) {
    std::fprintf(stderr, "%s: read %lld bytes, expected %zu\n", path.c_str(),
                 (long long)stream.gcount(), expected * sizeof(T));
    std::exit(2);
  }
  return data;
}

static void write_floats(const std::string &path, const std::vector<float> &data) {
  if (path.empty()) return;
  std::ofstream stream(path, std::ios::binary);
  stream.write(reinterpret_cast<const char *>(data.data()),
               (std::streamsize)(data.size() * sizeof(float)));
}

int main(int argc, char **argv) {
  Options options;
  if (!parse(argc, argv, options)) return 1;

  const long rows = options.rows;
  const long cols = options.cols;
  const int lanes = options.fp16 ? 8 : 4;
  if (cols % lanes != 0) {
    std::fprintf(stderr, "cols=%ld is not a multiple of %d; this ceiling kernel assumes vector loads\n",
                 cols, lanes);
    return 3;
  }
  const size_t count = (size_t)rows * cols;
  const size_t element = options.fp16 ? sizeof(__half) : sizeof(float);

  std::vector<float> host_x, host_w;
  std::vector<__half> host_x16, host_w16;
  if (options.fp16) {
    host_x16 = read_binary<__half>(options.x_path, count);
    host_w16 = read_binary<__half>(options.w_path, cols);
  } else {
    host_x = read_binary<float>(options.x_path, count);
    host_w = read_binary<float>(options.w_path, cols);
  }

  void *device_x = nullptr;
  void *device_w = nullptr;
  void *device_y = nullptr;
  CUDA_CHECK(cudaMalloc(&device_x, count * element));
  CUDA_CHECK(cudaMalloc(&device_w, cols * element));
  CUDA_CHECK(cudaMalloc(&device_y, count * element));
  CUDA_CHECK(cudaMemcpy(device_x, options.fp16 ? (const void *)host_x16.data() : (const void *)host_x.data(),
                        count * element, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_w, options.fp16 ? (const void *)host_w16.data() : (const void *)host_w.data(),
                        cols * element, cudaMemcpyHostToDevice));
  host_x.clear();
  host_x16.clear();

  const int vectors = (int)(cols / lanes);
  int block = std::max(32, std::min(256, ((vectors + 31) / 32) * 32));

  auto launch = [&]() {
    if (options.fp16) {
      rmsnorm_kernel<__half, 8><<<(int)rows, block>>>((const __half *)device_x, (const __half *)device_w,
                                                      (__half *)device_y, (int)cols, options.eps);
    } else {
      rmsnorm_kernel<float, 4><<<(int)rows, block>>>((const float *)device_x, (const float *)device_w,
                                                     (float *)device_y, (int)cols, options.eps);
    }
  };

  for (long i = 0; i < options.warmup; ++i) launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  std::vector<float> samples((size_t)options.reps);
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (long rep = 0; rep < options.reps; ++rep) {
    CUDA_CHECK(cudaEventRecord(start));
    launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    samples[(size_t)rep] = ms;
  }
  std::sort(samples.begin(), samples.end());
  const float ms_median = samples[samples.size() / 2];
  const float ms_min = samples.front();

  std::vector<float> produced(count);
  if (options.fp16) {
    std::vector<__half> raw(count);
    CUDA_CHECK(cudaMemcpy(raw.data(), device_y, count * sizeof(__half), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < count; ++i) produced[i] = __half2float(raw[i]);
  } else {
    CUDA_CHECK(cudaMemcpy(produced.data(), device_y, count * sizeof(float), cudaMemcpyDeviceToHost));
  }

  double max_abs = 0.0;
  double sum_sq_delta = 0.0;
  double sum_sq_reference = 0.0;
  const bool have_reference = !options.y_path.empty();
  if (have_reference) {
    std::vector<float> reference = read_binary<float>(options.y_path, count);
    for (size_t i = 0; i < count; ++i) {
      const double delta = std::fabs((double)produced[i] - (double)reference[i]);
      max_abs = std::max(max_abs, delta);
      sum_sq_delta += delta * delta;
      sum_sq_reference += (double)reference[i] * (double)reference[i];
    }
  }
  const double rel_l2 = have_reference
                            ? (sum_sq_reference > 0.0 ? std::sqrt(sum_sq_delta / sum_sq_reference) : 0.0)
                            : -1.0;

  write_floats(options.out_path, produced);

  cudaDeviceProp properties;
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  const double bytes = (2.0 * (double)count + (double)cols) * (double)element;
  const double gbps = bytes / (ms_median * 1e-3) / 1e9;

  std::printf("{\n");
  std::printf("  \"backend\": \"cuda_ceiling\",\n");
  std::printf("  \"key\": \"%s\",\n", options.key.c_str());
  std::printf("  \"kernel\": \"rmsnorm_block_per_row_vector%d\",\n", lanes);
  std::printf("  \"device\": \"%s\",\n", properties.name);
  std::printf("  \"compute_capability\": \"sm_%d%d\",\n", properties.major, properties.minor);
  std::printf("  \"driver_version\": %d,\n", properties.driverVersion);
  std::printf("  \"memory_clock_khz\": %d,\n", properties.memoryClockRate);
  std::printf("  \"memory_bus_bits\": %d,\n", properties.memoryBusWidth);
  std::printf("  \"multiprocessors\": %d,\n", properties.multiProcessorCount);
  std::printf("  \"cuda_version\": %d,\n", CUDA_VERSION);
  std::printf("  \"dtype\": \"%s\",\n", options.fp16 ? "fp16" : "fp32");
  std::printf("  \"rows\": %ld,\n", rows);
  std::printf("  \"cols\": %ld,\n", cols);
  std::printf("  \"threads_per_block\": %d,\n", block);
  std::printf("  \"eps\": %.9g,\n", options.eps);
  std::printf("  \"reps\": %ld,\n", options.reps);
  std::printf("  \"warmup\": %ld,\n", options.warmup);
  std::printf("  \"ms_median\": %.6f,\n", ms_median);
  std::printf("  \"ms_min\": %.6f,\n", ms_min);
  std::printf("  \"gb_s\": %.3f,\n", gbps);
  std::printf("  \"max_abs_err\": %.9g,\n", max_abs);
  std::printf("  \"rel_l2_err\": %.9g,\n", rel_l2);
  std::printf("  \"reference_checked\": %s\n", have_reference ? "true" : "false");
  std::printf("}\n");

  CUDA_CHECK(cudaFree(device_x));
  CUDA_CHECK(cudaFree(device_w));
  CUDA_CHECK(cudaFree(device_y));
  return 0;
}
