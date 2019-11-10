// Udacity HW 4
// Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============

   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

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
void BlellochScanImpl(cudaStream_t stream, unsigned int *const d_cdf,
                      int numBins, int block_size) {
  int grid_size = numBins / (2 * block_size);
  int share_size = 2 * block_size * sizeof(unsigned int);

  if (grid_size == 1) {
    UpSweep<<<grid_size, block_size, share_size, stream>>>(d_cdf, NULL);
    DownSweep<<<grid_size, block_size, share_size, stream>>>(d_cdf);
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

    UpSweep<<<grid_size, block_size, share_size, stream>>>(d_cdf, d_sums);
    DownSweep<<<grid_size, block_size, share_size, stream>>>(d_cdf);

    BlellochScanImpl(stream, d_sums, grid_size + padding, block_size);
    BlockSum<<<grid_size, block_size, 0, stream>>>(d_cdf, d_sums);

    checkCudaErrors(cudaFree(d_sums));
  }
}

// 4-bits histogram
__global__ void histogram(const unsigned *d_inputVals, size_t numElems,
                          unsigned nbits, unsigned offset, unsigned *count) {
  extern __shared__ unsigned block_count[];

  int tid = threadIdx.x;
  int gid = threadIdx.x + 2 * blockDim.x * blockIdx.x;

  unsigned nbins = 1 << nbits;
  if (tid < nbins) {
    block_count[tid] = 0;
  }
  __syncthreads();

  unsigned mask = (1 << nbits) - 1;
  if (gid < numElems) {
    unsigned idx = (d_inputVals[gid] >> offset) & mask;
    atomicAdd(&block_count[idx], 1);
  }

  gid += blockDim.x;
  if (gid < numElems) {
    unsigned idx = (d_inputVals[gid] >> offset) & mask;
    atomicAdd(&block_count[idx], 1);
  }
  __syncthreads();

  if (tid < nbins) {
    atomicAdd(&count[tid], block_count[tid]);
  }
}

__global__ void exclusive_scan_single(unsigned *data, int numElems) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    unsigned sum = 0;
    for (int i = 0; i < numElems; ++i) {
      unsigned temp = data[i];
      data[i] = sum;
      sum += temp;
    }
  }
}

__global__ void filter(const unsigned *d_inputVals, unsigned *d_outputIdx,
                       size_t numElems, unsigned nbits, unsigned offset,
                       unsigned bin) {
  int gid = threadIdx.x + 2 * blockDim.x * blockIdx.x;

  unsigned mask = (1 << nbits) - 1;
  if (gid < numElems) {
    unsigned val = (d_inputVals[gid] >> offset) & mask;
    d_outputIdx[gid] = (val == bin) ? 1 : 0;
  } else {
    d_outputIdx[gid] = 0;
  }

  gid += blockDim.x;
  if (gid < numElems) {
    unsigned val = (d_inputVals[gid] >> offset) & mask;
    d_outputIdx[gid] = (val == bin) ? 1 : 0;
  } else {
    d_outputIdx[gid] = 0;
  }
}

__global__ void apply(const unsigned *d_inputVals, unsigned *d_outputVals,
                      const unsigned *d_inputPos, unsigned *d_outputPos,
                      const unsigned *d_outputIdx, const unsigned *prefix_sums,
                      size_t numElems, unsigned nbits, unsigned offset,
                      unsigned bin) {
  int gid = threadIdx.x + 2 * blockDim.x * blockIdx.x;

  unsigned mask = (1 << nbits) - 1;
  unsigned prefix_sum = prefix_sums[bin];
  if (gid < numElems) {
    unsigned ival = d_inputVals[gid];
    unsigned val = (ival >> offset) & mask;
    if (val == bin) {
      unsigned idx = prefix_sum + d_outputIdx[gid];
      d_outputVals[idx] = ival;
      d_outputPos[idx] = d_inputPos[gid];
    }
  }

  gid += blockDim.x;
  if (gid < numElems) {
    unsigned ival = d_inputVals[gid];
    unsigned val = (ival >> offset) & mask;
    if (val == bin) {
      unsigned idx = prefix_sum + d_outputIdx[gid];
      d_outputVals[idx] = ival;
      d_outputPos[idx] = d_inputPos[gid];
    }
  }
}

__global__ void print(const unsigned *data, int numElems) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (int i = 0; i < numElems; ++i) {
      printf("%d ", data[i]);
    }
    printf("\n");
  }
}

void your_sort(unsigned int *const d_inputVals, unsigned int *const d_inputPos,
               unsigned int *const d_outputVals,
               unsigned int *const d_outputPos, const size_t numElems) {
  int nbits = 4;
  int nbins = 1 << nbits;
  int block_size = 256;
  int grid_size = (numElems + 2 * block_size - 1) / (2 * block_size);
  int padding = 2 * grid_size * block_size - numElems;

  cudaStream_t streams[nbins];
  unsigned *d_outputIdx[nbins];
  for (int i = 0; i < nbins; ++i) {
    // do not block with default stream
    checkCudaErrors(cudaStreamCreate(&streams[i]));

    // padding to 2 * block_size for Blelloch scan
    checkCudaErrors(
        cudaMalloc(&d_outputIdx[i], (numElems + padding) * sizeof(unsigned)));
  }

  unsigned *prefix_sums;
  checkCudaErrors(cudaMalloc(&prefix_sums, nbins * sizeof(unsigned)));

  // n-bits radix sort
  unsigned *p_inputVals = d_outputVals, *p_inputPos = d_outputPos;
  unsigned *p_outputVals = d_inputVals, *p_outputPos = d_inputPos;
  for (int offset = 0; offset < 32; offset += nbits) {
    // ping-pong input and output in each step
    std::swap(p_inputVals, p_outputVals);
    std::swap(p_inputPos, p_outputPos);

    checkCudaErrors(cudaMemset(prefix_sums, 0, nbins * sizeof(unsigned)));
    histogram<<<grid_size, block_size, block_size * sizeof(unsigned)>>>(
        p_inputVals, numElems, nbits, offset, prefix_sums);

    exclusive_scan_single<<<1, block_size>>>(prefix_sums, nbins);
    // std::cout << "offset: " << offset << ", prefix_sums: ";
    // print<<<1, block_size>>>(prefix_sums, nbins);
    // checkCudaErrors(cudaDeviceSynchronize());

    for (int bin = 0; bin < nbins; ++bin) {
      filter<<<grid_size, block_size, 0, streams[bin]>>>(
          p_inputVals, d_outputIdx[bin], numElems, nbits, offset, bin);

      BlellochScanImpl(streams[bin], d_outputIdx[bin], numElems + padding,
                       block_size);

      apply<<<grid_size, block_size, 0, streams[bin]>>>(
          p_inputVals, p_outputVals, p_inputPos, p_outputPos, d_outputIdx[bin],
          prefix_sums, numElems, nbits, offset, bin);
    }

    // synchronize all streams before next step
    for (int bin = 0; bin < nbins; ++bin) {
      std::cout << "cudaStreamSynchronize, offset: " << offset
                << ", stream: " << bin << std::endl;
      checkCudaErrors(cudaStreamSynchronize(streams[bin]));
    }
  }

  if (p_outputVals != d_outputVals) {
    checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals,
                               numElems * sizeof(unsigned),
                               cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos,
                               numElems * sizeof(unsigned),
                               cudaMemcpyDeviceToDevice));
  }
}

void radix_sort_test(const unsigned *h_inputVals, const unsigned *h_inputPos,
                     unsigned *h_outputVals, unsigned *h_outputPos,
                     int numElems) {
  size_t nbytes = numElems * sizeof(unsigned);
  unsigned *d_inputVals, *d_inputPos, *d_outputVals, *d_outputPos;

  checkCudaErrors(cudaMalloc(&d_inputVals, nbytes));
  checkCudaErrors(cudaMalloc(&d_inputPos, nbytes));

  checkCudaErrors(
      cudaMemcpy(d_inputVals, h_inputVals, nbytes, cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_inputPos, h_inputPos, nbytes, cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&d_outputVals, numElems * sizeof(unsigned)));
  checkCudaErrors(cudaMalloc(&d_outputPos, numElems * sizeof(unsigned)));

  your_sort(d_inputVals, d_inputPos, d_outputVals, d_outputPos, numElems);

  checkCudaErrors(
      cudaMemcpy(h_outputVals, d_outputVals, nbytes, cudaMemcpyDeviceToHost));
  checkCudaErrors(
      cudaMemcpy(h_outputPos, d_outputPos, nbytes, cudaMemcpyDeviceToHost));
}