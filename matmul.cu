#include <cuda_runtime.h>
#include <random>
#include <iostream>

__global__ void matmul(float* A, float* B, float* C, int K, int M, int N) {
    int x = threadIdx.x + (blockIdx.x * blockDim.x);
    int y = threadIdx.y + (blockIdx.y * blockDim.y);

    // A[K, M], B[M, N], C[K = y, N = x]
    if (y < K && x < N) {
        float sum = 0;
        for (int i = 0; i < M; ++i) {
            sum += A[y * M + i] * B[i * N + x];
        }
        C[y * N + x] = sum;
    }
}

int main(){
    int K = 300;
    int M = 400;
    int N = 500;

    float matrix1[K][M];
    float matrix2[M][N];

    // random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 9);

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < M; j++) {
            matrix1[i][j] = dist(gen);

        }
    }

    for (int i = 0;i<M;i++) {
        for (int j = 0;j<N;j++) {
            matrix2[i][j] = dist(gen);
        }
    }

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (K + threadsPerBlock.y - 1) / threadsPerBlock.y);

    float matrix3[K][N];

    float* A;
    float* B;
    float* C;
    cudaMalloc((void**)&A, sizeof(float) * K * M);
    cudaMalloc((void**)&B, sizeof(float) * M * N);
    cudaMalloc((void**)&C, sizeof(float) * K * N);
    cudaMemcpy(A, matrix1, sizeof(float) * K * M, cudaMemcpyHostToDevice);
    cudaMemcpy(B, matrix2, sizeof(float) * M * N, cudaMemcpyHostToDevice);

    matmul<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, K, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(matrix3, C, sizeof(float) * K * N, cudaMemcpyDeviceToHost);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int o = 0; o < M; ++o) {
                sum += matrix1[i][o] * matrix2[o][j];
            }
            if (sum != matrix3[i][j]) {
                std::cout << "Matmul wrong" << std::endl;
                exit(1);
            }
        }
    }

    std::cout << "Correct matmul" << std::endl;
    return 0;
}
