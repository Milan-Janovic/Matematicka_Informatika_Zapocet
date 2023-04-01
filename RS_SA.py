import random
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray
from numpy import sum as numpysum

# def population size -> ~multiplication time increase
_pop_size = 50


# Function 1: First Dejong function
def dejong1(x):
    return sum([xi ** 2 for xi in x])


def dejong2(x):
    # ensure that there are 2 coefficients
    assert len(x) >= 2
    x = asarray(x)
    return numpysum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


# Function 3: Schwefel function
def schwefel(x):
    n = len(x)
    return 418.9829 * n - sum([xi * np.sin(np.sqrt(abs(xi))) for xi in x])


def random_search(obj_func, bounds, max_iter, pop_size=_pop_size):
    # Initialize population with random solutions within the specified bounds
    pop = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(pop_size, bounds.shape[0]))

    # Evaluate fitness of each solution in the population
    fitness = [obj_func(ind) for ind in pop]

    # Record the best fitness and solution
    best_fitness = np.min(fitness)
    best_solution = pop[np.argmin(fitness)]

    # Perform the specified number of iterations
    for _ in range(max_iter):
        # Generate a new population with random solutions
        new_pop = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(pop_size, bounds.shape[0]))

        # Evaluate fitness of each solution in the new population
        fitness = [obj_func(ind) for ind in new_pop]

        # Update the best fitness and solution if a better one is found in the new population
        if np.min(fitness) < best_fitness:
            best_fitness = np.min(fitness)
            best_solution = new_pop[np.argmin(fitness)]

    # Return the best solution and fitness found
    return best_solution, best_fitness


def simulated_annealing(obj_func, bounds, max_iter, pop_size=_pop_size, temperature=100, cooling_rate=0.95):
    # Initialize the population and fitness arrays
    pop = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(pop_size, bounds.shape[0]))
    fitness = np.array([obj_func(ind) for ind in pop])
    # Set the initial best fitness and solution
    best_fitness = np.min(fitness)
    best_solution = pop[np.argmin(fitness)]
    # Initialize the acceptance probabilities array
    acceptance_probs = np.zeros(pop_size)

    # Loop through the specified number of iterations
    for i in range(max_iter):
        # Update the temperature
        temperature *= cooling_rate
        # Generate a candidate population
        candidate_pop = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(pop_size, bounds.shape[0]))
        # Evaluate the fitness of the candidate population
        candidate_fitness = np.array([obj_func(ind) for ind in candidate_pop])
        # Calculate the difference in fitness between the candidate and current populations
        delta_fitness = candidate_fitness - fitness

        # Determine which solutions in the candidate population are better than the current population
        better_solutions = delta_fitness < 0
        # Set the acceptance probabilities of the better solutions to 1
        acceptance_probs[better_solutions] = 1
        # Calculate the acceptance probabilities of the worse solutions
        acceptance_probs[~better_solutions] = np.exp(-delta_fitness[~better_solutions] / temperature)

        # Determine which solutions to accept based on their acceptance probabilities
        accept_solutions = np.random.rand(pop_size) < acceptance_probs
        # Update the current population and fitness with the accepted solutions
        pop[accept_solutions] = candidate_pop[accept_solutions]
        fitness[accept_solutions] = candidate_fitness[accept_solutions]

        # Update the best fitness and solution if a new one is found
        current_best_fitness = np.min(fitness)
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = pop[np.argmin(fitness)]

    # Return the best solution and fitness
    return best_solution, best_fitness


def plot_convergence(obj_func, bounds, title, filename, _max_iter, global_min, algo):
    best_fitness = float('inf')
    best_solution = None
    start_time = time.time()  # Get the current time
    best_solution_history = []
    best_solution_fitness_history = []
    for i in range(30):
        fitness_history = []
        solution_history = []
        if (i + 1) % 10 == 0 and i != 0:
            print("i = " + str(i + 1) + "/" + str(30))
        if algo == 'random_search':
            for j in range(1, _max_iter):
                best_solution_iter, fitness = random_search(obj_func, bounds, max_iter=_max_iter)
                fitness_history.append(fitness)
                solution_history.append(best_solution_iter)
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = best_solution_iter
        elif algo == 'simulated_annealing':
            for j in range(1, _max_iter):
                best_solution_iter, fitness = simulated_annealing(obj_func, bounds, max_iter=_max_iter)
                fitness_history.append(fitness)
                solution_history.append(best_solution_iter)
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = best_solution_iter
        best_solution_history.append(best_solution)
        best_solution_fitness_history.append(best_fitness)
        plt.plot(fitness_history, linewidth=0.5)
    print(f"Best solution for {title} after 30 runs: {best_solution}")
    print(f"Best fitness for {title} after 30 runs: {best_fitness}")

    end_time = time.time()  # Get the current time again
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print("Elapsed time:" + str(elapsed_time) + "seconds\n")

    plt.title(title, fontsize=30)
    plt.xlabel("Number of iterations", fontsize=20)
    plt.ylabel("Fitness", fontsize=20)
    plt.text(0.05, 0.95, f"Best solution: {best_solution}", transform=plt.gca().transAxes, va='top', fontsize=18)
    plt.text(0.05, 0.85, f"Best fitness: {best_fitness}", transform=plt.gca().transAxes, va='top', fontsize=18)
    plt.text(0.05, 0.8, f"Global minimum: {global_min}", transform=plt.gca().transAxes, va='top', fontsize=18)
    plt.text(10, 0.2, f"Elapsed time (30 runs) for {_max_iter} iterations each: {int(elapsed_time)} s; population size = {_pop_size}")
    # transform=plt.gca().transAxes, va='top',fontsize=18)
    plt.savefig(f"/Users/milanjanovic/Desktop/{filename}.png")
    plt.clf()  # clear the figure to avoid overlapping plots
