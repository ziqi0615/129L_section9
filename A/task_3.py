#!/usr/bin/env python

from os.path import isfile
from numpy import pi, cos, sin, linspace, exp
from numpy.random import seed as srand, normal, rand
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# A: Function Definitions for Optimization
def potential_function(theta):
    return theta**4 - 8 * theta**2 - 2 * cos(4 * pi * theta)

def potential_gradient(theta):
    return 4 * theta**3 - 16 * theta + 8 * pi * sin(4 * pi * theta)

def initialize_plot():
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$H(\theta)$')
    ax.set_xlim(-3, 3)
    theta_values = linspace(-3, 3, 301)
    ax.plot(theta_values, potential_function(theta_values))
    return fig, ax

def perform_gradient_descent(initial_theta, learning_rates, output_filename):
    if isfile(output_filename):
        return
    fig, ax = initialize_plot()
    marker, = ax.plot([], [], c='r', marker='o')
    time_label = ax.text(0.05, 0.95, '', transform=ax.transAxes)
    theta = initial_theta

    def initialize_animation():
        marker.set_data([theta], [potential_function(theta)])
        return marker,

    def update_animation(rate):
        nonlocal theta
        theta -= rate * potential_gradient(theta)
        marker.set_data([theta], [potential_function(theta)])
        time_label.set_text(f'{rate=:.4f}')
        return marker, time_label

    animation = FuncAnimation(fig, update_animation, frames=learning_rates, init_func=initialize_animation, blit=True)
    animation.save(output_filename, writer='ffmpeg', fps=10)

perform_gradient_descent(-1, 1 / linspace(50, 200, 31), 'gradient_descent_-1.mkv')
perform_gradient_descent(0.5, 1 / linspace(50, 200, 31), 'gradient_descent_0.5.mkv')
perform_gradient_descent(3, 1 / linspace(50, 200, 31), 'gradient_descent_3.mkv')

# B: Metropolis-Hastings Algorithm
def metropolis_hastings_sampler(initial_theta, inverse_temperature, output_filename, steps=30, random_seed=1108):
    if isfile(output_filename):
        return
    srand(random_seed)
    fig, ax = initialize_plot()
    marker, = ax.plot([], [], c='r', marker='o')
    time_label = ax.text(0.05, 0.95, '', transform=ax.transAxes)
    theta = initial_theta
    potential_value = potential_function(theta)
    iteration_count = 0

    def initialize_animation():
        marker.set_data([theta], [potential_value])
        return marker,

    def update_animation(_):
        nonlocal theta, potential_value, iteration_count
        while True:
            new_theta = theta + normal()
            new_potential = potential_function(new_theta)
            acceptance_ratio = exp(-inverse_temperature * (new_potential - potential_value))
            iteration_count += 1
            if acceptance_ratio > 1 or rand() < acceptance_ratio:
                theta, potential_value = new_theta, new_potential
                break
        marker.set_data([theta], [potential_value])
        time_label.set_text(f'{iteration_count=}')
        return marker, time_label

    animation = FuncAnimation(fig, update_animation, frames=steps, init_func=initialize_animation, blit=True)
    animation.save(output_filename, writer='ffmpeg', fps=10)

metropolis_hastings_sampler(-1, 1, 'metropolis_hastings_-1_1.mkv')
metropolis_hastings_sampler(0.5, 1, 'metropolis_hastings_0.5_1.mkv')
metropolis_hastings_sampler(3, 1, 'metropolis_hastings_3_1.mkv')
metropolis_hastings_sampler(-1, 3, 'metropolis_hastings_-1_3.mkv')
metropolis_hastings_sampler(0.5, 3, 'metropolis_hastings_0.5_3.mkv')
metropolis_hastings_sampler(3, 3, 'metropolis_hastings_3_3.mkv')

# C: Simulated Annealing
def simulated_annealing(initial_theta, temperature_schedule, output_filename, random_seed=1108):
    if isfile(output_filename):
        return
    srand(random_seed)
    fig, ax = initialize_plot()
    marker, = ax.plot([], [], c='r', marker='o')
    time_label = ax.text(0.05, 0.95, '', transform=ax.transAxes)
    theta = initial_theta
    potential_value = potential_function(theta)

    def initialize_animation():
        marker.set_data([theta], [potential_value])
        return marker,

    def update_animation(current_temperature):
        nonlocal theta, potential_value
        new_theta = theta + normal()
        new_potential = potential_function(new_theta)
        acceptance_ratio = exp(-current_temperature * (new_potential - potential_value))
        if acceptance_ratio > 1 or rand() < acceptance_ratio:
            theta, potential_value = new_theta, new_potential
        marker.set_data([theta], [potential_value])
        time_label.set_text(f'{current_temperature=:.4f}')
        return marker, time_label

    animation = FuncAnimation(fig, update_animation, frames=temperature_schedule, init_func=initialize_animation, blit=True)
    animation.save(output_filename, writer='ffmpeg', fps=10)

simulated_annealing(-1, linspace(1, 2, 30), 'annealing_-1.mkv')
simulated_annealing(0.5, linspace(1, 2, 30), 'annealing_0.5.mkv')
simulated_annealing(3, linspace(1, 2, 30), 'annealing_3.mkv')
