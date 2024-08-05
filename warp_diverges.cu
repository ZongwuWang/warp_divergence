#include <stdio.h>
#include <cuda.h>
#include <mma.h>

using namespace nvcuda;

// 矩阵维度定义
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define N 10240
#define THREADS_PER_BLOCK 32

// Kernel with warp divergence
__global__ void warp_divergent_kernel_homo(int *a, int *val) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index % 2 == 0) {
        for (int i = 0; i < 20000; ++i) {
            a[index] += *val;
			// __nanosleep(10000);
        }
    } else {
        for (int i = 0; i < 20000; ++i) {
            a[index] -= *val;
			// __nanosleep(10000);
        }		
    }
}

// Kernel with warp divergence
__global__ void warp_divergent_kernel_hete(int *a, half *b, half *c, float *d, int *val) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index % 2 == 0) {
        for (int i = 0; i < 20000; ++i) {
            a[index] += *val;
			// __nanosleep(10000);
        }
    } 
		for (int i = 0; i < 10000; ++i) {
			// 声明用于 MMA 操作的片段
			wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
			wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> c_frag;
			wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
			// wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> d_frag;

			// 初始化累加器片段
			wmma::fill_fragment(acc_frag, 0.0f);
			
			// 加载矩阵 A 和 B 到片段中
			wmma::load_matrix_sync(b_frag, b, WMMA_K);
			wmma::load_matrix_sync(c_frag, c, WMMA_K);

			// 执行矩阵乘法和累加操作
			wmma::mma_sync(acc_frag, b_frag, c_frag, acc_frag);

			// 将结果写回到全局内存
			wmma::store_matrix_sync(d, acc_frag, WMMA_K, wmma::mem_row_major);
		}		
    
}

__global__ void warp_divergent_kernel_hete2(int *a, half *b, half *c, float *d, int *val) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // if (index % 2 == 0) {
    //     for (int i = 0; i < 20000; ++i) {
    //         a[index] += *val;
	// 		// __nanosleep(10000);
    //     }
    // } 
		for (int i = 0; i < 10000; ++i) {
			// 声明用于 MMA 操作的片段
			wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
			wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> c_frag;
			wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
			// wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> d_frag;

			// 初始化累加器片段
			wmma::fill_fragment(acc_frag, 0.0f);
			
			// 加载矩阵 A 和 B 到片段中
			wmma::load_matrix_sync(b_frag, b, WMMA_K);
			wmma::load_matrix_sync(c_frag, c, WMMA_K);

			// 执行矩阵乘法和累加操作
			wmma::mma_sync(acc_frag, b_frag, c_frag, acc_frag);

			// 利用tensor core计算期间并行进行cuda core的操作
			a[index] += *val;
			if (index < 16) {
				for (int j = 0; j < 4; ++j) {
					a[index * 2] += a[index * 2 + 1];
				}
			}

			// 将结果写回到全局内存
			wmma::store_matrix_sync(d, acc_frag, WMMA_K, wmma::mem_row_major);
		}		
    
}

// Kernel with warp divergence
__global__ void tensor_core_kernel(half *b, half *c, float *d) {
	for (int i = 0; i < 10000; ++i) {
		// 声明用于 MMA 操作的片段
		wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
		wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> c_frag;
		wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
		// wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> d_frag;

		// 初始化累加器片段
		wmma::fill_fragment(acc_frag, 0.0f);
		
		// 加载矩阵 A 和 B 到片段中
		wmma::load_matrix_sync(b_frag, b, WMMA_K);
		wmma::load_matrix_sync(c_frag, c, WMMA_K);

		// 执行矩阵乘法和累加操作
		wmma::mma_sync(acc_frag, b_frag, c_frag, acc_frag);

		// 将结果写回到全局内存
		wmma::store_matrix_sync(d, acc_frag, WMMA_K, wmma::mem_row_major);
	}
}

// Kernel without warp divergence
__global__ void cuda_core_kernel(int *a, int *val) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = 0; i < 20000; ++i) {
        a[index] += *val;
		// __nanosleep(10000);
    }
}

int main(void) {
    int *a;
	half *b, *c;
    float *d;
	int *val;
	int host_val = 1;
    cudaMallocManaged(&a, N*sizeof(int));
    cudaMallocManaged(&b, WMMA_M * WMMA_K * sizeof(half));
    cudaMallocManaged(&c, WMMA_K * WMMA_N * sizeof(half));
    cudaMallocManaged(&d, WMMA_M * WMMA_N * sizeof(float));
	cudaMallocManaged(&val, sizeof(int));
	*val = host_val;

    // Initialize array
    for (int i = 0; i < N; ++i) {
        a[i] = 0;
    }

    // Launch kernels and measure execution time
    cudaEvent_t start, stop;
    float time1, time2, time3, time4, time5;

	for (int i = 0; i < 50; i++) {

		// Warp coherent kernel
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		cuda_core_kernel<<<1, THREADS_PER_BLOCK>>>(a, val);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time1, start, stop);
		printf("Time for the cuda core kernel without warp divergence: %f ms\n", time1);

				// Warp coherent kernel
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		tensor_core_kernel<<<1, THREADS_PER_BLOCK>>>(b, c, d);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time2, start, stop);
		printf("Time for the tensor core kernel without warp divergence: %f ms\n", time2);

		
		// Tensor core kernel
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		warp_divergent_kernel_homo<<<1, THREADS_PER_BLOCK>>>(a, val);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time3, start, stop);
		printf("Time for the cuda core (homogeneous) kernel with warp divergence: %f ms\n", time3);
				
		// Tensor core kernel
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		warp_divergent_kernel_hete<<<1, THREADS_PER_BLOCK>>>(a, b, c, d, val);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time4, start, stop);
		printf("Time for the cuda core and tensor core (heterogeneous) kernel with warp divergence: %f ms\n", time4);

		// Tensor core kernel
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		warp_divergent_kernel_hete2<<<1, THREADS_PER_BLOCK>>>(a, b, c, d, val);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time5, start, stop);
		printf("Time for the cuda core and tensor core (heterogeneous) kernel2 with warp divergence: %f ms\n", time5);

		printf("============================================\n");
		// printf("Without warp divergence speedup compared to with warp divergence: %f\n", time1 / (time2 + time3));
	}
    

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(d);
	cudaFree(val);

    return 0;
}
