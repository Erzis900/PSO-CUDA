#include <iostream>
#include <limits>
#include <fstream>
#include <chrono>
#include <vector>
#include "../include/kernels.cuh"

float calc_std_var(std::vector<float> times, float avg)
{
    float sum = 0;
    for (auto &x : times)
    {
        sum += std::pow((x - avg), 2);
    }

    return std::sqrt(sum / times.size());
}

std::vector<Particle> initParticles(int funcIndex)
{
    std::vector<Particle> particles;
    for (int i = 0; i < swarmSize; i++)
    {
        Particle p;
        p.x = static_cast<float>(rand()) / RAND_MAX * (HI - LO) + LO;
        p.y = static_cast<float>(rand()) / RAND_MAX * (HI - LO) + LO;
        p.vX = 0;
        p.vY = 0;
        p.pBestX = p.x;
        p.pBestY = p.y;

        switch (funcIndex) {
            case 0:
                // p.pBest = Func_Booth(p.x, p.y);
                p.pBest = pow(p.x + 2 * p.y - 7, 2) + pow(2 * p.x + p.y - 5, 2);
                break;
            // case 1:
            //     p.pBest = Func_Sphere(p.x, p.y);
            //     break;
            // case 2:
            //     p.pBest = Func_Rosenbrock(p.x, p.y);
            //     break;
            // case 3:
            //     p.pBest = Func_Rastrigin(p.x, p.y);
            //     break;
            default:
                p.pBest = INFINITY;
                break;
        }
        particles.push_back(p);
    }
    return particles;
}

int main(int argc, char* argv[]) {
    // int deviceId;
    // cudaGetDevice(&deviceId);

    // int numberOfSMs;
    // cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    //std::cout << numberOfSMs << std::endl;

    float time_whole_n = 0;
    std::vector<float> xn_whole;

    float duration = 0;

    std::ofstream csvFile("../data.csv");
    csvFile << "Iteration,X,Y,gBestX,gBestY,gBest,avgPBest\n";
    
    std::vector<float> h_positions(swarmSize * 6);

    float totalGBest, totalGX, totalGY = 0.0f;

    int funcIndex = 3;

    // case 0:
    //     p.pBest = Func_Booth(p.x, p.y);
    //     break;
    // case 1:
    //     p.pBest = Func_Sphere(p.x, p.y);
    //     break;
    // case 2:
    //     p.pBest = Func_Rosenbrock(p.x, p.y);
    //     break;
    // case 3:
    //     p.pBest = Func_Rastrigin(p.x, p.y);
    //     break;

    curandState* rngState;
    float gBestX, gBestY, gBest = std::numeric_limits<float>::max();
    float* d_positions;
    float* d_avgPBest;

    std::vector<float> gBests;

    int numberOfRuns = std::atoi(argv[1]);
    std::cout << "Number of runs: " << numberOfRuns << std::endl;
    std::cout << "Swarm size: " << swarmSize << std::endl;
    std::cout << "Running..." << std::endl;

    for (int run = 0; run < numberOfRuns + 1; run++) {
        auto t1 = std::chrono::high_resolution_clock::now();

        Particle* d_particles;
        // std::vector<Particle> h_particles = initParticles(funcIndex);
        size_t particlesSize = sizeof(Particle) * swarmSize;

        cudaMalloc(&rngState, sizeof(curandState) * swarmSize);

        cudaMalloc(&d_particles, particlesSize);
        // cudaMemcpy(d_particles, h_particles.data(), particlesSize, cudaMemcpyHostToDevice);

        // cudaMalloc(&d_avgPBest, sizeof(float));

        Wrapper::WInitRNG(rngState, clock());
        // cudaMalloc(&d_positions, sizeof(float) * swarmSize * 6);

        gBestX, gBestY, gBest = std::numeric_limits<float>::max();
        Wrapper::WInitParticles(d_particles, rngState, funcIndex);

        for (int i = 0; i < maxIterations; i++) {
            // cudaMemset(d_avgPBest, 0, sizeof(float));

            // cudaMemPrefetchAsync(h_positions.data(), sizeof(float) * swarmSize * 6, deviceId);
            Wrapper::WUpdate(d_particles, rngState, swarmSize, gBestX, gBestY, funcIndex);
            // Wrapper::WCalculateAveragePBest(d_particles, d_avgPBest, swarmSize);
            Wrapper::WUpdateBestIndex(d_particles, swarmSize, &gBest, &gBestX, &gBestY, i, d_positions);
            cudaDeviceSynchronize();

            // std::cout << i << " gBest: " << gBest << std::endl;

            // float h_avgPBest = 0;
            // cudaMemcpy(&h_avgPBest, d_avgPBest, sizeof(float), cudaMemcpyDeviceToHost);

            // h_avgPBest /= swarmSize; 

            //std::cout << "Average pBest: " << h_avgPBest << std::endl;

            // cudaMemcpy(h_positions.data(), d_positions, sizeof(float) * swarmSize * 6, cudaMemcpyDeviceToHost);

            // for (int j = 0; j < swarmSize; j++) {
            //     csvFile << h_positions[j * 6 + 0] << ","
            //             << h_positions[j * 6 + 1] << ","
            //             << h_positions[j * 6 + 2] << ","
            //             << h_positions[j * 6 + 3] << ","
            //             << h_positions[j * 6 + 4] << ","
            //             << h_positions[j * 6 + 5] << ","
            //             << h_avgPBest << "\n";
            // }
        }
        totalGBest += gBest;
        totalGX += gBestX;
        totalGY += gBestY;
        gBests.push_back(gBest);

        // cudaFree(d_particles);
        // cudaFree(rngState);
        // cudaFree(d_positions);
        // cudaFree(d_avgPBest);

        auto t2 = std::chrono::high_resolution_clock::now();
        if(run != 0)
        {
            // xn.push_back(time_n);
            // time_n = 0;   
            // std::cout << gBest << std::endl;

            duration = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
            time_whole_n += duration;

            xn_whole.push_back(duration);

            cudaFree(d_particles);
            cudaFree(rngState);
            cudaFree(d_positions);
            cudaFree(d_avgPBest);
        }
    }
    // float sum_x = 0;
    // for (auto &x : xn)
    // {
    //     sum_x += x;
    //     // std::cout << x << ",";
    // }
    // std::cout << std::endl;

    // for (auto &x : xn_whole)
    // {
    //     std::cout << x << ",";
    // }

    std::cout << std::endl;

    // float avg_time = totalKernels / numberOfRuns;
    // float avg_time = sum_x / numberOfRuns;
    // float std_var = calc_std_var(xn, avg_time);

    float avg_whole_run = time_whole_n / numberOfRuns;
    float std_whole_var = calc_std_var(xn_whole, avg_whole_run);

    // std::cout << "Average optimization time: " << avg_whole_run << " ms" << std::endl;
    std::cout << "Average optimization time: " << avg_whole_run << " microseconds" << std::endl;
    std::cout << "Standard deviation of optimization time: " << std_whole_var << std::endl;
    // std::cout << "Average kernel execution time: " << avg_time << " microseconds" << std::endl;
    // std::cout << "Standard kernel deviation: " << std_var << std::endl;
    std::cout << "Final gBest: " << gBest << std::endl;
    std::cout << "Final position: (" << gBestX << ", " << gBestY << ")" << std::endl;

    float averageGBest = totalGBest / numberOfRuns;
    std::cout << "Average gBest: " << averageGBest << std::endl;
    std::cout << "gBest std var: " << calc_std_var(gBests, averageGBest) << std::endl;

    float averageGX = totalGX / numberOfRuns;
    float averageGY = totalGY / numberOfRuns;
    std::cout << "Average position: (" << averageGX << ", " << averageGY << ")" << std::endl;

    std::ofstream timeFile("../time_gpu.csv", std::ios::app);
    timeFile << swarmSize << "," << avg_whole_run << "," << std_whole_var << "\n";

    timeFile.close();

    csvFile.close();

    return 0;
}