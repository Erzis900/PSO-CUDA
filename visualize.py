import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--iterations', default=None)
args = parser.parse_args()

# Booth function
def func(x, y):
    return (x + 2 * y - 7)**2 + (2 * x + y - 5)**2

particle_data = pd.read_csv('data.csv')

x_range = np.linspace(-10, 10, 500)
y_range = np.linspace(-10, 10, 500)
x, y = np.meshgrid(x_range, y_range)
z = func(x, y)

x_min = x.ravel()[z.argmin()]
y_min = y.ravel()[z.argmin()]

if not os.path.exists('img'):
    os.mkdir('img')

if args.iterations == None:
    max_iter = len(particle_data['Iteration'].unique())
else:
    max_iter = int(args.iterations)

print("Generating " + str(max_iter) + " plots...")

for iteration in range(0, max_iter):
    print("Generating plot " + str(iteration + 1))
    plt.figure()
    plt.imshow(z, extent=[-10, 10, -10, 10], cmap='viridis', alpha=0.8)

    plt.colorbar(label='Function Value')
    plt.plot(x_min, y_min, 'ko', markersize=6,
             label='Global Minimum')
    plt.contour(x, y, z, levels=30, colors='white', linewidths=0.8)
    plt.title(f'Contour Plot at Iteration {iteration + 1}', fontsize=15)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    
    iteration_data = particle_data[particle_data['Iteration'] == iteration]
    plt.scatter(iteration_data['X'], iteration_data['Y'], color='blue', alpha=0.7, label=f'Particles\' positions')
    
    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    plt.savefig(f'img/contour_plot_iteration_{iteration + 1}.png', bbox_inches='tight')
    plt.close()
    
print("Plots saved to 'img' directory")