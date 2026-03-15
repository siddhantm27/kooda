#include <random>
#include <iostream>

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

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            matrix3[i * N + j] = 0;
            for (int k = 0; k < M; k++) {
                matrix3[i * N + j] += matrix1[i * M + k] * matrix2[k * N + j];
            }
        }
    }

    delete[] matrix1;
    delete[] matrix2;
    delete[] matrix3;

    return 0;
}