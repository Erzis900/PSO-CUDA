#include <iostream>
#include <limits>
#include <fstream>
#include <chrono>
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

    std::ofstream csvFile("../data.csv");
    csvFile << "Iteration,X,Y\n";

    auto t1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < maxIterations; i++) {
        Wrapper::WUpdate(d_particles, rngState, swarmSize, gBestX, gBestY);
        Wrapper::WUpdateBestIndex(d_particles, swarmSize, &gBest, &gBestX, &gBestY, i, d_positions);

        float* h_positions = new float[swarmSize * 3]; 
        cudaMemcpy(h_positions, d_positions, sizeof(float) * swarmSize * 3, cudaMemcpyDeviceToHost);

        for (int j = 0; j < swarmSize; j++) {
            csvFile << h_positions[j * 3 + 0] << "," << h_positions[j * 3 + 1] << "," << h_positions[j * 3 + 2] << "\n";
        }

        delete[] h_positions;
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    std::cout << "Calculated in: " << duration.count() << " ms" << std::endl;

    std::cout << "Final gBest: " << gBest << std::endl;
    std::cout << "Final position: (" << gBestX << ", " << gBestY << ")" << std::endl;

    cudaFree(d_particles);
    cudaFree(rngState);
    cudaFree(d_positions);

    csvFile.close();

    return 0;
}