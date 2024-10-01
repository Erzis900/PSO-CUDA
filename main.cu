#include <iostream>
#include <vector>
#include <limits>
#include <math.h>

const int LO = -5;
const int HI = 5;

float RandFloat(int LO, int HI)
{
    return LO + static_cast<float>(rand()) / ( static_cast<float>(RAND_MAX / (HI-LO)));
}

double Func(double x, double y) 
{
    return pow(x + 2 * y - 7, 2) + pow(2 * x + y - 5, 2);
}

class Particle
{
    public:
        Particle()
        {
            x = RandFloat(LO, HI);
            y = RandFloat(LO, HI);

            vX = 0;
            vY = 0;

            pBestX = x;
            pBestY = y;

            pBest = Func(x, y);
        }

        void Update(float gBestX, float gBestY, float w, float c1, float c2)
        {
            r1 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            r2 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

            vX = w * vX + c1 * r1 * (pBestX - x) + c2 * r2 * (gBestX - x);
            vY = w * vY + c1 * r1 * (pBestY - y) + c2 * r2 * (gBestY - y);

            x += vX;
            y += vY;

            pBestNew = Func(x, y);

            if (pBestNew < pBest)
            {
                pBestX = x;
                pBestY = y;
                pBest = pBestNew;
            }
        }

        float getX() { return x; }
        float getY() { return y; }
        float getPBest() { return pBest; }
        
    private:
        float x, y;
        float vX, vY;

        float pBestX, pBestY;
        float pBest;

        float pBestNew;

        float r1, r2;
};

int main()
{
    srand(time(0));

    int swarmSize = 20;
    int maxIterations = 30;

    int w = 0.5;
    int c1 = 1;
    int c2 = 2;

    std::vector<Particle> swarm(swarmSize);
    float gBestX = swarm[0].getX();
    float gBestY = swarm[0].getY();
    float gBest = std::numeric_limits<float>::max();

    for (int i = 0; i < maxIterations; i++) 
    {
        for (int j = 0; j < swarmSize; j++)
        {
            swarm[j].Update(gBestX, gBestY, w, c1, c2);

            if (swarm[j].getPBest() < gBest)
            {
                gBestX = swarm[j].getX();
                gBestY = swarm[j].getY();
                gBest = swarm[j].getPBest();
            }
        }
        std::cout << "Iteration " << i + 1 << " gBest: " << gBest << std::endl;
    }

    std::cout << "Final gBest: " << gBest << std::endl;
    std::cout << "Final position: (" << gBestX << ", " << gBestY << ")" << std::endl;

    return 0;
}