#include <iostream>
#include <limits>
#include <fstream>
#include <chrono>
#include <vector>
#include "../include/kernels.cuh"

int main() {
    int deviceId;
    cudaGetDevice(&deviceId);

    int numberOfSMs;
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    //std::cout << numberOfSMs << std::endl;

    curandState* rngState;
    Particle* d_particles;
    float gBestX, gBestY, gBest = std::numeric_limits<float>::max();
    float* d_positions;

    cudaMalloc(&rngState, sizeof(curandState) * swarmSize);
    cudaMalloc(&d_particles, sizeof(Particle) * swarmSize);
    cudaMalloc(&d_positions, sizeof(float) * swarmSize * 6);

    Wrapper::WInitRNG(rngState, clock());
    Wrapper::WInitParticles(d_particles, rngState);

    int totalKernels = 0;

    std::ofstream csvFile("../data.csv");
    csvFile << "Iteration,X,Y,gBestX,gBestY,gBest\n";
    
    std::vector<float> h_positions(swarmSize * 6);

    auto t1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < maxIterations; i++) {
        auto kernelStart = std::chrono::high_resolution_clock::now();

        Wrapper::WUpdate(d_particles, rngState, swarmSize, gBestX, gBestY);
        Wrapper::WUpdateBestIndex(d_particles, swarmSize, &gBest, &gBestX, &gBestY, i, d_positions);

        auto kernelEnd = std::chrono::high_resolution_clock::now();
        int kernelDuration = static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(kernelEnd - kernelStart).count());

        totalKernels += kernelDuration;

        cudaMemcpy(h_positions.data(), d_positions, sizeof(float) * swarmSize * 6, cudaMemcpyDeviceToHost);

        for (int j = 0; j < swarmSize; j++) {
            csvFile << h_positions[j * 6 + 0] << ","
                    << h_positions[j * 6 + 1] << ","
                    << h_positions[j * 6 + 2] << ","
                    << h_positions[j * 6 + 3] << ","
                    << h_positions[j * 6 + 4] << ","
                    << h_positions[j * 6 + 5] << "\n";
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    //std::cout << "Whole loop with cudaMemcpy executed in: " << duration.count() << " ms" << std::endl;
    std::cout << "Kernels executed in: " << totalKernels << " microseconds" << std::endl;
    std::cout << "Final gBest: " << gBest << std::endl;
    std::cout << "Final position: (" << gBestX << ", " << gBestY << ")" << std::endl;

    cudaFree(d_particles);
    cudaFree(rngState);
    cudaFree(d_positions);
    csvFile.close();

    return 0;
}