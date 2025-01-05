import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--iterations', type=int, help='Number of iterations to generate plots for')
args = parser.parse_args()

# Load particle data from CSV
data = pd.read_csv('data.csv')

# Get unique iterations
all_iterations = data['Iteration'].unique()

# Determine which iterations to plot
if args.iterations is not None:
    iterations = np.unique(all_iterations)[:args.iterations]
else:
    iterations = all_iterations

# Create a directory for saving plots
output_dir = 'particle_progression_plots'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Function to generate the surface for contour plotting
def func_booth(x, y):
    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2

def func_rastrigin(x, y):
    A = 10
    return A * 2 + (x * x) + (y * y) - A * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))

# Prepare grid for function values
x_range = np.linspace(-10, 10, 500)
y_range = np.linspace(-10, 10, 500)
x, y = np.meshgrid(x_range, y_range)
z = func_rastrigin(x, y)

# Create the contour levels
contour_levels = np.linspace(z.min(), z.max(), 10)

global_min_x = 0
global_min_y = 0

# Iterate over the specified iterations
for iteration in iterations:
    iteration_data = data[data['Iteration'] == iteration]
    
    plt.figure(figsize=(10, 8))
    
    # Create a contour plot
    plt.contourf(x, y, z, levels=contour_levels, alpha=0.7)
    plt.colorbar(label='Function Value')

    # Plot each particle's position
    for index in range(iteration_data.shape[0]):
        plt.scatter(iteration_data['X'].iloc[index], iteration_data['Y'].iloc[index], color='yellow', s=100, alpha=0.6, label='Particles' if index == 0 else "")
    
    # Plot the global best position
    
    # Plot the global minimum
    plt.scatter(global_min_x, global_min_y, color='white', marker='o', s=100, label='Global Minimum')
    plt.scatter(iteration_data['gBestX'].iloc[-1], iteration_data['gBestY'].iloc[-1], color='red', marker='x', s=100, label='Global Best Position')

    plt.title(f'Particle Positions at Iteration {iteration}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.xlim([-10, 10])  # Adjust according to your function's range
    plt.ylim([-10, 10])
    plt.legend()
    # plt.grid()

    print("Saving plot " + str(iteration))

    # Save the figure
    # plt.savefig(f'{output_dir}/iteration_{iteration}.eps', format='eps', dpi=1200)
    plt.savefig(f'{output_dir}/iteration_{iteration}.png')
    plt.close()

print("Particle progression plots saved in 'particle_progression_plots' directory.")
