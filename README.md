# PSO-CUDA
 Particle Swarm Optimization algorithm in CUDA

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install numpy, matplotlib and pandas.

```bash
pip3 install numpy matplotlib pandas
```

## Compilation

The CUDA file can be compiled using the NVCC compiler.
```bash
nvcc main.cu -o main
```

## Running
You can run the program by executing the binary file.
```bash
./main
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