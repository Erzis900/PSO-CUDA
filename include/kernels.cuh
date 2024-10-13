#include <curand_kernel.h>

const int LO = -10;
const int HI = 10;
const int swarmSize = 1000;
const int maxIterations = 100;

const float w = 0.5;
const float c1 = 1.5;
const float c2 = 1.5;

struct Particle {
    float x, y;
    float vX, vY;
    float pBestX, pBestY;
    float pBest;
};

namespace Wrapper {
    void WInitRNG(curandState* state, unsigned long long seed);
    void WInitParticles(Particle* d_particles, curandState* state, int funcIndex);
    void WUpdate(Particle* d_particles, curandState* state, int swarmSize, float gBestX, float gBestY, int funcIndex);
    void WUpdateBestIndex(Particle* d_particles, int swarmSize, float* gBest, float* gBestX, float* gBestY, int iteration, float* d_positions);
    //void WUpdateCSV(Particle* d_particles, int swarmSize, int iteration, float* d_positions);
}