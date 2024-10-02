#include <iostream>
#include <limits>
#include <curand_kernel.h>
#include <fstream>

const int LO = -10;
const int HI = 10;

const int swarmSize = 20;
const int maxIterations = 30;

const float w = 0.5;
const float c1 = 1.5;
const float c2 = 1.5;

// Booth function
// Global minimum = Func(1, 3) = 0
__device__ float Func(float x, float y) {
    return pow(x + 2 * y - 7, 2) + pow(2 * x + y - 5, 2);
}

struct Particle {
    float x, y;
    float vX, vY;
    float pBestX, pBestY;
    float pBest;
};

__global__ void InitRNG(curandState* state, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void InitParticles(Particle* d_particles, curandState* state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < swarmSize) {
        Particle p;
        p.x = LO + (HI - LO) * curand_uniform(&state[idx]);
        p.y = LO + (HI - LO) * curand_uniform(&state[idx]);
        p.vX = 0;
        p.vY = 0;
        p.pBestX = p.x;
        p.pBestY = p.y;
        p.pBest = Func(p.x, p.y);
        d_particles[idx] = p;
    }
}

__global__ void Update(Particle* d_particles, curandState* state, int swarmSize, float gBestX, float gBestY) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < swarmSize) {
        float r1 = curand_uniform(&state[idx]);
        float r2 = curand_uniform(&state[idx]);

        Particle& p = d_particles[idx];
        p.vX = w * p.vX + c1 * r1 * (p.pBestX - p.x) + c2 * r2 * (gBestX - p.x);
        p.vY = w * p.vY + c1 * r1 * (p.pBestY - p.y) + c2 * r2 * (gBestY - p.y);

        p.x += p.vX;
        p.y += p.vY;

        float pBestNew = Func(p.x, p.y);
        if (pBestNew < p.pBest) {
            p.pBestX = p.x;
            p.pBestY = p.y;
            p.pBest = pBestNew;
        }
    }
}

__global__ void UpdateBestIndex(Particle* d_particles, int swarmSize, float* gBest, float* gBestX, float* gBestY, int iteration, float* d_positions) {
    for (int i = 0; i < swarmSize; i++) {
        if (d_particles[i].pBest < *gBest) {
            *gBestX = d_particles[i].x;
            *gBestY = d_particles[i].y;
            *gBest = d_particles[i].pBest;
        }

        d_positions[i * 3 + 0] = iteration;
        d_positions[i * 3 + 1] = d_particles[i].x;
        d_positions[i * 3 + 2] = d_particles[i].y;
    }
}

int main() {
    curandState* rngState;
    Particle* d_particles;
    float gBestX, gBestY, gBest = std::numeric_limits<float>::max();
    float* d_positions;

    cudaMalloc(&rngState, sizeof(curandState) * swarmSize);
    cudaMalloc(&d_particles, sizeof(Particle) * swarmSize);
    cudaMalloc(&d_positions, sizeof(float) * swarmSize * 3);

    int blockSize = 256;
    int gridSize = (swarmSize + blockSize - 1) / blockSize;

    InitRNG<<<gridSize, blockSize>>>(rngState, clock());
    InitParticles<<<gridSize, blockSize>>>(d_particles, rngState);

    std::ofstream csvFile("data.csv");
    csvFile << "Iteration,X,Y\n";

    for (int i = 0; i < maxIterations; i++) {
        Update<<<gridSize, blockSize>>>(d_particles, rngState, swarmSize, gBestX, gBestY);
        UpdateBestIndex<<<gridSize, blockSize>>>(d_particles, swarmSize, &gBest, &gBestX, &gBestY, i, d_positions);

        float* h_positions = new float[swarmSize * 3];
        cudaMemcpy(h_positions, d_positions, sizeof(float) * swarmSize * 3, cudaMemcpyDeviceToHost);

        for (int j = 0; j < swarmSize; j++) {
            csvFile << h_positions[j * 3 + 0] << "," << h_positions[j * 3 + 1] << "," << h_positions[j * 3 + 2] << "\n";
        }

        delete[] h_positions;
    }

    std::cout << "Final gBest: " << gBest << std::endl;
    std::cout << "Final position: (" << gBestX << ", " << gBestY << ")" << std::endl;

    cudaFree(d_particles);
    cudaFree(rngState);
    cudaFree(d_positions);

    csvFile.close();

    return 0;
}