/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.


  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <limits>

template <typename reduce_op>
__global__ void reduce_kernel(const float *const d_logLuminance,
                              float *d_output, const size_t numElems,
                              reduce_op op) {
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;

  extern __shared__ float s_data[];

  if (gid < numElems) {
    s_data[tid] = d_logLuminance[gid];
  }
  __syncthreads();

  int offset = blockIdx.x * blockDim.x;
  int nsize =
      (offset + blockDim.x < numElems) ? blockDim.x : (numElems - offset);
  while (nsize > 1) {
    int len = (nsize + 1) / 2;
    if (tid < nsize / 2) {
      s_data[tid] = op(s_data[tid], s_data[tid + len]);
    }
    nsize = len;
    __syncthreads();
  }

  if (tid == 0) {
    d_output[blockIdx.x] = s_data[0];
  }
}

template <typename reduce_op>
float reduce(const float *const d_logLuminance, int numElems, float init,
             const reduce_op &op) {
  int block_size = 1024;
  int grid_size = (numElems + block_size - 1) / block_size;

  float *d_output;
  checkCudaErrors(cudaMalloc(&d_output, grid_size * sizeof(float)));

  reduce_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(
      d_logLuminance, d_output, numElems, op);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
  printf("first round, grid_size: %d\n", grid_size);

  numElems = grid_size;
  for (; numElems >= block_size; numElems = grid_size) {
    grid_size = (numElems + block_size - 1) / block_size;
    printf("grid_size: %d\n", grid_size);

    float *d_input = d_output;
    checkCudaErrors(cudaMalloc(&d_output, grid_size * sizeof(float)));

    reduce_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(
        d_input, d_output, numElems, op);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaFree(d_input));
  }

  printf("grid_size: %d, numElems: %d\n", grid_size, numElems);
  float *h_output = new float[numElems];
  checkCudaErrors(cudaMemcpy(h_output, d_output, numElems * sizeof(float),
                             cudaMemcpyDeviceToHost));
  cudaFree(d_output);

  float res = init;
  for (int i = 0; i < numElems; ++i) {
    res = op(res, h_output[i]);
  }
  delete[] h_output;

  return res;
}

__global__ void histogram(const float *const d_logLuminance, size_t numElems,
                          unsigned int *const d_cdf, size_t numBins,
                          float min_logLum, float max_logLum) {
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid < numElems) {
    int bin = (d_logLuminance[gid] - min_logLum) / (max_logLum - min_logLum) *
              numBins;
    atomicAdd(&d_cdf[bin], 1);
  }
}

__global__ void HS_scan_kernel(unsigned int *d_cdf, unsigned int *d_out,
                               int numBins, int stride) {
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid < numBins) {
    if (gid - stride < 0) {
      d_out[gid] = d_cdf[gid];
    } else {
      d_out[gid] = d_cdf[gid] + d_cdf[gid - stride];
    }
  }
}

// Hillis & Steele inclusive scan
void HS_scan_impl(unsigned int **d_cdf, unsigned int **d_out, int numBins,
                  int block_size) {
  for (int stride = 1; stride < numBins; stride <<= 1) {
    // swith input and output every step
    if (stride != 1) {
      unsigned int *temp = *d_cdf;
      *d_cdf = *d_out;
      *d_out = temp;
    }

    int grid_size = (numBins + block_size - 1) / block_size;
    HS_scan_kernel<<<grid_size, block_size>>>(*d_cdf, *d_out, numBins, stride);

    // TODO: should just copy d_cdf to d_out to reduce threads num in next step?
  }
}

void HS_scan(unsigned int *const h_cdf, int numBins, int block_size) {
  // need an out array to avoid data race
  unsigned int *d_cdf, *d_out;
  checkCudaErrors(cudaMalloc(&d_cdf, numBins * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_out, numBins * sizeof(unsigned int)));

  checkCudaErrors(cudaMemcpy(d_cdf, h_cdf, numBins * sizeof(unsigned int),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_out, h_cdf, numBins * sizeof(unsigned int),
                             cudaMemcpyHostToDevice));

  HS_scan_impl(&d_cdf, &d_out, numBins, block_size);
  checkCudaErrors(cudaMemcpy(h_cdf, d_out, numBins * sizeof(unsigned int),
                             cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(d_cdf));
  checkCudaErrors(cudaFree(d_out));
}

__global__ void UpSweep(unsigned int *const d_cdf, unsigned int *const d_sums) {
  extern __shared__ unsigned int temp[];

  int tid = threadIdx.x;
  int gid = threadIdx.x + 2 * blockIdx.x * blockDim.x;
  temp[tid] = d_cdf[gid];
  temp[tid + blockDim.x] = d_cdf[gid + blockDim.x];
  __syncthreads();

  for (int stride = 1; stride < 2 * blockDim.x; stride <<= 1) {
    int idx = (tid + 1) * stride * 2 - 1;
    if (idx < 2 * blockDim.x) {
      temp[idx] = temp[idx - stride] + temp[idx];
    }

    __syncthreads();
  }

  d_cdf[gid] = temp[tid];
  d_cdf[gid + blockDim.x] = temp[tid + blockDim.x];

  if (tid == 0 && d_sums != NULL) {
    d_sums[blockIdx.x] = temp[2 * blockDim.x - 1];
  }
}

__global__ void DownSweep(unsigned int *const d_cdf) {
  extern __shared__ unsigned int temp[];

  int tid = threadIdx.x;
  int gid = threadIdx.x + 2 * blockIdx.x * blockDim.x;
  temp[tid] = d_cdf[gid];
  temp[tid + blockDim.x] = d_cdf[gid + blockDim.x];
  __syncthreads();

  temp[2 * blockDim.x - 1] = 0;

  for (int stride = blockDim.x; stride >= 1; stride >>= 1) {
    int idx = (tid + 1) * stride * 2 - 1;
    if (idx < 2 * blockDim.x) {
      int t = temp[idx - stride];
      temp[idx - stride] = temp[idx];
      temp[idx] += t;
    }

    __syncthreads();
  }

  d_cdf[gid] = temp[tid];
  d_cdf[gid + blockDim.x] = temp[tid + blockDim.x];
}

__global__ void BlockSum(unsigned int *const d_cdf,
                         unsigned int *const d_sums) {
  int gid = threadIdx.x + 2 * blockIdx.x * blockDim.x;
  d_cdf[gid] += d_sums[blockIdx.x];
  d_cdf[gid + blockDim.x] += d_sums[blockIdx.x];
}

// REQURES: d_cdf padding to 2 * block_size
void Blelloch_scan_impl(unsigned int *const d_cdf, int numBins,
                        int block_size) {
  int grid_size = numBins / (2 * block_size);
  int share_size = 2 * block_size * sizeof(unsigned int);

  if (grid_size == 1) {
    UpSweep<<<grid_size, block_size, share_size>>>(d_cdf, NULL);
    DownSweep<<<grid_size, block_size, share_size>>>(d_cdf);
  } else {
    unsigned int *d_sums;
    // padding block sums for futher Blelloch scan
    int grid_size_ = (grid_size + 2 * block_size - 1) / (2 * block_size);
    int padding = 2 * grid_size_ * block_size - grid_size;
    checkCudaErrors(
        cudaMalloc(&d_sums, (grid_size + padding) * sizeof(unsigned int)));

    if (padding > 0) {
      checkCudaErrors(
          cudaMemset(d_sums + grid_size, 0, padding * sizeof(unsigned int)));
    }

    UpSweep<<<grid_size, block_size, share_size>>>(d_cdf, d_sums);
    DownSweep<<<grid_size, block_size, share_size>>>(d_cdf);

    Blelloch_scan_impl(d_sums, grid_size + padding, block_size);
    BlockSum<<<grid_size, block_size>>>(d_cdf, d_sums);

    checkCudaErrors(cudaFree(d_sums));
  }
}

// Blelloch exclusive scan
// reference: https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html
void Blelloch_scan(unsigned int *const h_cdf, int numBins, int block_size) {
  unsigned int *d_cdf;
  // each thread block will handle two data block
  int grid_size = (numBins + 2 * block_size - 1) / (2 * block_size);
  int padding = 2 * grid_size * block_size - numBins;

  checkCudaErrors(
      cudaMalloc(&d_cdf, (numBins + padding) * sizeof(unsigned int)));
  checkCudaErrors(cudaMemcpy(d_cdf, h_cdf, numBins * sizeof(unsigned int),
                             cudaMemcpyHostToDevice));

  // padding last block
  if (padding > 0) {
    checkCudaErrors(
        cudaMemset(d_cdf + numBins, 0, padding * sizeof(unsigned int)));
  }

  Blelloch_scan_impl(d_cdf, numBins + padding, block_size);

  checkCudaErrors(cudaMemcpy(h_cdf, d_cdf, numBins * sizeof(unsigned int),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_cdf));
}

void your_histogram_and_prefixsum(const float *const d_logLuminance,
                                  unsigned int *const d_cdf, float &min_logLum,
                                  float &max_logLum, const size_t numRows,
                                  const size_t numCols, const size_t numBins) {
  // TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
  min_logLum = reduce(
      d_logLuminance, numRows * numCols, std::numeric_limits<float>::max(),
      [] __host__ __device__(float a, float b) { return (a < b) ? a : b; });

  printf("minimum: %f\n", min_logLum);

  max_logLum = reduce(
      d_logLuminance, numRows * numCols, std::numeric_limits<float>::min(),
      [] __host__ __device__(float a, float b) { return (a < b) ? b : a; });

  printf("maximum: %f\n", max_logLum);

  int numElems = numRows * numCols;
  int block_size = 1024;
  int grid_size = (numElems + block_size - 1) / block_size;

  histogram<<<grid_size, block_size>>>(d_logLuminance, numElems, d_cdf, numBins,
                                       min_logLum, max_logLum);

  Blelloch_scan_impl(d_cdf, numBins, block_size);
}