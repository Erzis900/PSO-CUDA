#include "../include/kernels.cuh"

int blockSize = 1024;
int gridSize = (swarmSize + blockSize - 1) / blockSize;

// Booth function
// Global minimum = Func(1, 3) = 0
__device__ float Func_Booth(float x, float y) {
    return pow(x + 2 * y - 7, 2) + pow(2 * x + y - 5, 2);
}

// Sphere function
// Global minimum = Func(0, 0) = 0
__device__ float Func_Sphere(float x, float y) {
    return pow(x, 2) + pow(y, 2);
}

// Rosenbrock function
// Global minimum = Func(1, 1) = 0
__device__ float Func_Rosenbrock(float x, float y) {
    const float a = 1.0f;
    const float b = 100.0f;
    return pow(a - x, 2) + b * pow(y - x * x, 2);
}

// Rastrigin function
// Global minimum = Func(0, 0) = 0
__device__ float Func_Rastrigin(float x, float y) {
    const float A = 10.0f;
    return A * 2 + (x * x) + (y * y) - A * (cos(2 * M_PI * x) + cos(2 * M_PI * y));
}

__global__ void InitRNG(curandState* state, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void InitParticles(Particle* d_particles, curandState* state, int funcIndex) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int particleNo = idx; particleNo < swarmSize; particleNo += stride) {
        Particle p;
        p.x = LO + (HI - LO) * curand_uniform(&state[particleNo]);
        p.y = LO + (HI - LO) * curand_uniform(&state[particleNo]);
        p.vX = 0;
        p.vY = 0;
        p.pBestX = p.x;
        p.pBestY = p.y;

        switch (funcIndex) {
            case 0:
                p.pBest = Func_Booth(p.x, p.y);
                break;
            case 1:
                p.pBest = Func_Sphere(p.x, p.y);
                break;
            case 2:
                p.pBest = Func_Rosenbrock(p.x, p.y);
                break;
            case 3:
                p.pBest = Func_Rastrigin(p.x, p.y);
                break;
            default:
                p.pBest = INFINITY;
                break;
        }
        d_particles[particleNo] = p;
    }
}

__global__ void Update(Particle* d_particles, curandState* state, int swarmSize, float gBestX, float gBestY, int funcIndex) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int particleNo = idx; particleNo < swarmSize; particleNo += stride) {
        float r1 = curand_uniform(&state[particleNo]);
        float r2 = curand_uniform(&state[particleNo]);

        Particle& p = d_particles[particleNo];

        p.vX = w * p.vX + c1 * r1 * (p.pBestX - p.x) + c2 * r2 * (gBestX - p.x);
        p.vY = w * p.vY + c1 * r1 * (p.pBestY - p.y) + c2 * r2 * (gBestY - p.y);

        p.x += p.vX;
        p.y += p.vY;

        float pBestNew;
        switch (funcIndex) {
            case 0:
                pBestNew = Func_Booth(p.x, p.y);
                break;
            case 1:
                pBestNew = Func_Sphere(p.x, p.y);
                break;
            case 2:
                pBestNew = Func_Rosenbrock(p.x, p.y);
                break;
            case 3:
                pBestNew = Func_Rastrigin(p.x, p.y);
                break;
            default:
                pBestNew = INFINITY;
                break;
        }

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

        // d_positions[i * 6 + 0] = iteration + 1;
        // d_positions[i * 6 + 1] = d_particles[i].x;
        // d_positions[i * 6 + 2] = d_particles[i].y;
        // d_positions[i * 6 + 3] = *gBestX;
        // d_positions[i * 6 + 4] = *gBestY;
        // d_positions[i * 6 + 5] = *gBest;
    }
}

__global__ void CalculateAveragePBest(Particle* d_particles, float* avgPBest, int swarmSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread accumulates its pBest
    float localSum = 0.0f;
    if (idx < swarmSize) {
        localSum = d_particles[idx].pBest;
    }

    // Use shared memory to sum pBest values within a block
    __shared__ float sharedSum[1024]; // Adjust based on your block size
    sharedSum[threadIdx.x] = localSum;
    __syncthreads();

    // Reduction to sum the shared pBest values
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            sharedSum[threadIdx.x] += sharedSum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Write the block's total to global memory
    if (threadIdx.x == 0) {
        atomicAdd(avgPBest, sharedSum[0]);
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

    void WInitParticles(Particle* d_particles, curandState* state, int funcIndex) {
        InitParticles<<<gridSize, blockSize>>>(d_particles, state, funcIndex);
    }

    void WUpdate(Particle* d_particles, curandState* state, int swarmSize, float gBestX, float gBestY, int funcIndex) {
        Update<<<gridSize, blockSize>>>(d_particles, state, swarmSize, gBestX, gBestY, funcIndex);
    }

    void WUpdateBestIndex(Particle* d_particles, int swarmSize, float* gBest, float* gBestX, float* gBestY, int iteration, float* d_positions) {
        UpdateBestIndex<<<1,1>>>(d_particles, swarmSize, gBest, gBestX, gBestY, iteration, d_positions);
    }

    void WCalculateAveragePBest(Particle* d_particles, float* d_avgPBest, int swarmSize) {
        CalculateAveragePBest<<<gridSize, blockSize>>>(d_particles, d_avgPBest, swarmSize);
    }

    /*void WUpdateCSV(Particle* d_particles, int swarmSize, int iteration, float* d_positions) {
        UpdateCSV<<<1, 1>>>(d_particles, swarmSize, iteration, d_positions);
    }*/
}