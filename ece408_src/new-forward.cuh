
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

constexpr int TILE_WIDTH = 8;

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K) {

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    #define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
    #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
    #define k4d(i3,i2,i1,i0) k[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]

/*
    int W_grid = (int) ceil(W_out / TILE_WIDTH * 1.0);
    //  int H_grid = ceil(H_out / TILE_WIDTH * 1.0);
    int n, m, h, w, c, p, q;
    n = blockIdx.x;
    m = blockIdx.y;
    h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y;
    w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;

    float acc = 0.0;
    for (c = 0; c < C; c++) {
      for (p = 0; p < K; p++) {
        for (q = 0; q < K; q++) {
//        if (h+p < H && w+q < W)
            acc += x4d(n, c, h+p, w+q) * k4d(m, c, p, q);
          }
        }
    }

    y4d(n, m, h, w) = acc;
*/

    // Implementation starts at p. 15, Chapter 16 of the textbook
    int n, m, h0, w0, h_base, w_base, h, w;

    int X_tile_width = TILE_WIDTH + K - 1;

    // shared memory array
    extern __shared__ float shmem[];

    // shared memory for input X and weights K
    float* X_shared = &shmem[0];
    float* W_shared = &shmem[X_tile_width * X_tile_width];

    n = blockIdx.x;
    m = blockIdx.y;
    h0 = threadIdx.x;
    w0 = threadIdx.y;

    // Textbook says TILE_SIZE, maybe a typo. I'm going to try TILE_WIDTH
    h_base = blockIdx.z / ((H_out-1)/TILE_WIDTH+1) * (TILE_WIDTH);
    w_base = blockIdx.z % ((W_out-1)/TILE_WIDTH+1) * (TILE_WIDTH);

    h = h_base + h0;
    w = w_base + w0;

    float acc = 0;

    int c, i, j, p, q;

    for(c = 0; c < C; c++)
    {
      if((h0 < K) && (w0 < K))
      {
        W_shared[h0*K+w0] = k4d(m, c, h0, w0);
      }

      __syncthreads();

      for(i = h; i < h_base + X_tile_width; i += TILE_WIDTH)
      {
        for(j = w; j < w_base + X_tile_width; j += TILE_WIDTH)
        {
            X_shared[(i - h_base)*X_tile_width + (j - w_base)] = x4d(n, c, i, j);
        }
      }

      __syncthreads();

      for(p = 0; p < K; p++)
      {
        for(q = 0; q < K; q++)
        {
            acc += X_shared[(h0 + p)*X_tile_width + (w0 + q)] * W_shared[p*K+q];
        }
      }
      __syncthreads();

    }

    y4d(n, m, h, w) = acc;

    #undef y4d
    #undef x4d
    #undef k4d

}

// This function is called by new-inl.h
// Any code you write should be executed by this function
template<>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {
    // Use mxnet's CHECK_EQ to do assertions.

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    cudaStream_t s = y.stream_->stream_;

    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    int H_out = H - K + 1;
    int W_out = W - K + 1;

    int W_grid = ((W_out - 1) / TILE_WIDTH+1);
    int H_grid = ((H_out - 1) / TILE_WIDTH+1);

    int Z = H_grid * W_grid;

    // Set the kernel dimensions
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(B, M, Z);

    // Declare shared memory variable (FOR SECOND KERNEL)
    size_t shmem_size = sizeof(float)*((TILE_WIDTH+K-1)*(TILE_WIDTH+K-1)+K*K);
    forward_kernel<<<gridDim, blockDim, shmem_size, s>>>(y.dptr_,x.dptr_,w.dptr_,B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}


template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    assert( 0 && "No forward implementation for other datatypes needed for ECE408");
}


}
}

#endif
