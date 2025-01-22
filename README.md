# PSO-CUDA
 Particle Swarm Optimization algorithm in CUDA.
 Built on Ubuntu 24.04.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install numpy, matplotlib and pandas.

```bash
pip3 install numpy matplotlib pandas
```

## Building

The project can be built using CMake.
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

## Running
You can run the programs by executing their binary files. The binary files take the number of runs as argument. If you want to run the script once, execute:

CPU implementation:
```bash
./pso_cpu 1
```
GPU implementation
```bash
./pso_cuda 1
```
The program will generate a data.csv file which can be used to visualize particles' position in each iteration.

## Visualization

You can run
```bash
python3 visualize.py
```
to generate images for maximum number of iterations specified in `main.cu`. Images will be stored in `img`.

If you wish to generate less images you can use the `-i` flag. For example
```bash
python3 visualize.py -i 10
```
will only generate 10 images for first 10 iterations.

## License

[MIT](https://choosealicense.com/licenses/mit/)