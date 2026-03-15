#include <cuda_runtime.h>
#include <random>
#include <iostream>

__global__ void matmul(float* A, float* B, float* C, int K, int M, int N) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < K && y < N) {
        float sum = 0;
        for (int i = 0; i < M; i++) {
            sum += A[x * M + i] * B[i * N + y];
        }
        C[x * N + y] = sum;
    }
}

int main(int argc, char* argv[]){
    int K, M, N;
    K = M = N = std::atoi(argv[1]);

    float* matrix1 = new float[K * M];
    float* matrix2 = new float[M * N];
    float* matrix3 = new float[K * N];

    // random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 9);

    for(int i=0;i<K*M;i++) matrix1[i] = dist(gen);
    for(int i=0;i<M*N;i++) matrix2[i] = dist(gen);

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

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

    delete[] matrix1;
    delete[] matrix2;
    delete[] matrix3;

    return 0;
}
