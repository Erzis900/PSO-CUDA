#include "../include/kernels.cuh"

int blockSize = 128;
int gridSize = (swarmSize + blockSize - 1) / blockSize;

// Booth function
// Global minimum = Func(1, 3) = 0
__device__ float Func(float x, float y) {
    return pow(x + 2 * y - 7, 2) + pow(2 * x + y - 5, 2);
}

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

        d_positions[i * 6 + 0] = iteration;
        d_positions[i * 6 + 1] = d_particles[i].x;
        d_positions[i * 6 + 2] = d_particles[i].y;
        d_positions[i * 6 + 3] = *gBestX;
        d_positions[i * 6 + 4] = *gBestY;
        d_positions[i * 6 + 5] = *gBest;
    }
}

/*__global__ void UpdateCSV(Particle* d_particles, int swarmSize, int iteration, float* d_positions) {
    for (int i = 0; i < swarmSize; i++) {
        d_positions[i * 3 + 0] = iteration;
        d_positions[i * 3 + 1] = d_particles[i].x;
        d_positions[i * 3 + 2] = d_particles[i].y;
    }
}*/

namespace Wrapper {
	void WInitRNG(curandState* state, unsigned long long seed) {
		InitRNG<<<gridSize, blockSize>>>(state, seed);
	}

    void WInitParticles(Particle* d_particles, curandState* state) {
        InitParticles<<<gridSize, blockSize>>>(d_particles, state);
    }

    void WUpdate(Particle* d_particles, curandState* state, int swarmSize, float gBestX, float gBestY) {
        Update<<<gridSize, blockSize>>>(d_particles, state, swarmSize, gBestX, gBestY);
    }

    void WUpdateBestIndex(Particle* d_particles, int swarmSize, float* gBest, float* gBestX, float* gBestY, int iteration, float* d_positions) {
        UpdateBestIndex<<<1,1>>>(d_particles, swarmSize, gBest, gBestX, gBestY, iteration, d_positions);
    }

    /*void WUpdateCSV(Particle* d_particles, int swarmSize, int iteration, float* d_positions) {
        UpdateCSV<<<1, 1>>>(d_particles, swarmSize, iteration, d_positions);
    }*/
}