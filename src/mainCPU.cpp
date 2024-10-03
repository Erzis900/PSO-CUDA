#include <iostream>
#include <vector>
#include <limits>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cmath>
#include <fstream>

const int LO = -10;
const int HI = 10;
const int swarmSize = 500;
const int maxIterations = 1000;

const float w = 0.5f;
const float c1 = 1.5f;
const float c2 = 1.5f;

struct Particle {
    float x, y;
    float vX, vY;
    float pBestX, pBestY;
    float pBest;

    Particle() {
        x = static_cast<float>(rand()) / RAND_MAX * (HI - LO) + LO;
        y = static_cast<float>(rand()) / RAND_MAX * (HI - LO) + LO;
        vX = 0.0f;
        vY = 0.0f;
        pBestX = x;
        pBestY = y;
        pBest = Func(x, y);
    }

    static float Func(float x, float y) {
        return pow(x + 2 * y - 7, 2) + pow(2 * x + y - 5, 2);
    }

    void Update(float gBestX, float gBestY) {
        float r1 = static_cast<float>(rand()) / RAND_MAX;
        float r2 = static_cast<float>(rand()) / RAND_MAX;

        vX = w * vX + c1 * r1 * (pBestX - x) + c2 * r2 * (gBestX - x);
        vY = w * vY + c1 * r1 * (pBestY - y) + c2 * r2 * (gBestY - y);

        x += vX;
        y += vY;

        float pBestNew = Func(x, y);
        if (pBestNew < pBest) {
            pBestX = x;
            pBestY = y;
            pBest = pBestNew;
        }
    }
};

int main() {
    std::srand(static_cast<unsigned int>(std::time(0)));

    std::vector<Particle> particles(swarmSize);
    float gBestX, gBestY, gBest = std::numeric_limits<float>::max();

    std::ofstream csvFile("../data.csv");
    csvFile << "Iteration,X,Y\n";

    auto t1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < maxIterations; i++) {
        for (auto& p : particles) {
            p.Update(gBestX, gBestY);
            if (p.pBest < gBest) {
                gBestX = p.pBestX;
                gBestY = p.pBestY;
                gBest = p.pBest;
            }
        }

        for (const auto& p : particles) {
            csvFile << i << "," << p.x << "," << p.y << "\n";
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    std::cout << "Calculated in: " << duration.count() << " ms" << std::endl;
    std::cout << "Final gBest: " << gBest << std::endl;
    std::cout << "Final position: (" << gBestX << ", " << gBestY << ")" << std::endl;

    csvFile.close();

    return 0;
}
