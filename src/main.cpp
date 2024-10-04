#include <iostream>
#include <limits>
#include <fstream>
#include <chrono>
#include <vector>
#include "../include/kernels.cuh"

int main() {
    curandState* rngState;
    Particle* d_particles;
    float gBestX, gBestY, gBest = std::numeric_limits<float>::max();
    float* d_positions;

    cudaMalloc(&rngState, sizeof(curandState) * swarmSize);
    cudaMalloc(&d_particles, sizeof(Particle) * swarmSize);
    cudaMalloc(&d_positions, sizeof(float) * swarmSize * 3);

    Wrapper::WInitRNG(rngState, clock());
    Wrapper::WInitParticles(d_particles, rngState);

    int totalKernels = 0;

    std::vector<std::vector<float>> all_positions(maxIterations, std::vector<float>(swarmSize * 3));
    
    auto t1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < maxIterations; i++) {
        auto kernelStart = std::chrono::high_resolution_clock::now();

        Wrapper::WUpdate(d_particles, rngState, swarmSize, gBestX, gBestY);
        Wrapper::WUpdateBestIndex(d_particles, swarmSize, &gBest, &gBestX, &gBestY, i, d_positions);

        auto kernelEnd = std::chrono::high_resolution_clock::now();
        int kernelDuration = static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(kernelEnd - kernelStart).count());

        totalKernels += kernelDuration;

        cudaMemcpy(all_positions[i].data(), d_positions, sizeof(float) * swarmSize * 3, cudaMemcpyDeviceToHost);
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    std::ofstream csvFile("../data.csv");
    csvFile << "Iteration,X,Y\n";
    
    for (int i = 0; i < maxIterations; i++) {
        for (int j = 0; j < swarmSize; j++) {
            csvFile
            << all_positions[i][j * 3 + 0] << ","
            << all_positions[i][j * 3 + 1] << "," 
            << all_positions[i][j * 3 + 2] << "\n";
        }
    }

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