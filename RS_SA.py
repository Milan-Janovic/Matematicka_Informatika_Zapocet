import random
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray
from numpy import sum as numpysum
from progress.bar import Bar

# def population size -> ~multiplication time increase
_pop_size_RS = 1
_pop_size_SA = 1
number_of_metropolis_calls = 10

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


def random_search(obj_func, bounds, max_iter, pop_size=_pop_size_RS):
    fitness_progress = []

    # Initialize population with random solutions within the specified bounds
    pop = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(pop_size, bounds.shape[0]))

    # Evaluate fitness of each solution in the population
    fitness = [obj_func(ind) for ind in pop]

    # Record the best fitness and solution
    best_fitness = np.min(fitness)
    best_solution = pop[np.argmin(fitness)]

    # Calculate radius
    #radius = 0.1 * np.abs(bounds[:, 1] - bounds[:, 0])

    # Perform the specified number of iterations
    for i in range(max_iter):
        # Generate a new population with random solutions within 10% radius of the best solution
        #new_pop = np.random.uniform(np.maximum(bounds[:, 0], best_solution - radius),
                                    #np.minimum(bounds[:, 1], best_solution + radius),
                                    #size=(pop_size, bounds.shape[0]))

        # Generate a new population with random solutions
        # Changed back to random from above code that generated new candidates within the 10% radius
        # Not sure if this should be implemented for RS and SA or just SA
        # As the RS should be RANDOM ??? -> maybe confirm with Mr. Senkerik
        new_pop = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(pop_size, bounds.shape[0]))

        # Evaluate fitness of each solution in the new population
        fitness = [obj_func(ind) for ind in new_pop]

        # Update the best fitness and solution if a better one is found in the new population
        if np.min(fitness) < best_fitness:
            best_fitness = np.min(fitness)
            best_solution = new_pop[np.argmin(fitness)]

        fitness_progress.append(best_fitness)

    # Return the best solution and fitness found
    return best_solution, best_fitness, fitness_progress


def simulated_annealing(obj_func, bounds, max_iter, temperature=100, cooling_rate=0.99):
    fitness_progress = []

    # Initialize the population and fitness arrays
    pop = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(1, bounds.shape[0]))
    fitness = np.array([obj_func(pop[0])])
    # Set the initial best fitness and solution
    best_fitness = np.min(fitness)
    best_solution = pop[np.argmin(fitness)]

    # Calculate radius
    radius = 0.1 * np.abs(bounds[:, 1] - bounds[:, 0])

    # Loop through the specified number of iterations
    for i in range(max_iter):
        # Update the temperature
        temperature *= cooling_rate

        # Loop through 10 candidates
        for j in range(number_of_metropolis_calls):
            # Generate a candidate solution within 10% radius of the best solution
            candidate = np.random.uniform(np.maximum(bounds[:, 0], best_solution - radius),
                                          np.minimum(bounds[:, 1], best_solution + radius),
                                          size=(_pop_size_SA, bounds.shape[0]))
            candidate_fitness = obj_func(candidate[0])

            # Calculate the difference in fitness between the candidate and current solution
            delta_fitness = candidate_fitness - fitness[0]

            # If the candidate solution has a lower fitness, accept it
            if delta_fitness < 0:
                pop[0] = candidate[0]
                fitness[0] = candidate_fitness

                # Update the best fitness and solution if necessary
                if candidate_fitness < best_fitness:
                    best_fitness = candidate_fitness
                    best_solution = candidate[0]

            # If the candidate solution has a higher fitness, accept it with a probability determined by the
            # Metropolis-Hastings criterion
            else:
                acceptance_prob = np.exp(-delta_fitness / temperature)
                if np.random.rand() < acceptance_prob:
                    pop[0] = candidate[0]
                    fitness[0] = candidate_fitness

            fitness_progress.append(best_fitness)

        # Update the best fitness and solution if necessary (in case a better solution was found in the inner loop)
        # Not sure about this -> not visible improvement & sometimes re-adds same solutions ?!
        current_best_fitness = np.min(fitness)
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = pop[np.argmin(fitness)]

    # Return the best solution and fitness
    return best_solution, best_fitness, fitness_progress


def plot_convergence(obj_func, bounds, title, filename, _max_iter, global_min, algo):
    best_fitness = float('inf')
    best_solution = None
    start_time = time.time()  # Get the current time
    best_solution_history = []
    best_solution_fitness_history = []
    fitness_progress_avg_30_runs = []
    bar = Bar('Iteration : ', max=30)
    for i in range(30):
        bar.next()
        if algo == 'random_search':
            best_solution_iter, best_fitness_iter, fitness_progress = random_search(obj_func, bounds,
                                                                                    max_iter=_max_iter)
        elif algo == 'simulated_annealing':
            best_solution_iter, best_fitness_iter, fitness_progress = simulated_annealing(obj_func, bounds,
                                                                                          max_iter=_max_iter)
        if best_fitness_iter < best_fitness:
            best_fitness = best_fitness_iter
            best_solution = best_solution_iter

        best_solution_history.append(best_solution_iter)
        best_solution_fitness_history.append(best_fitness_iter)
        fitness_progress_avg_30_runs.append(fitness_progress)
        plt.plot(fitness_progress, linewidth=0.5)

    bar.finish()

    print(f"Best solution for {title} after 30 runs: {best_solution}")
    print(f"Best fitness for {title} after 30 runs: {best_fitness}")

    end_time = time.time()  # Get the current time again
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print("Elapsed time:" + str(elapsed_time) + "seconds\n")

    plt.title(title, fontsize=30)
    plt.xlabel("Number of iterations", fontsize=20)
    plt.ylabel("Fitness", fontsize=20)
    plt.text(0.05, 0.95, f"Best solution: {best_solution}", transform=plt.gca().transAxes, va='top', fontsize=16)
    plt.text(0.05, 0.85, f"Best fitness: {best_fitness}", transform=plt.gca().transAxes, va='top', fontsize=16)
    plt.text(0.05, 0.8, f"Global minimum: {global_min}", transform=plt.gca().transAxes, va='top', fontsize=16)
    plt.savefig(f"./Fitness_Graphs/{filename}.png")
    plt.clf()  # clear the figure to avoid overlapping plots

    # Calculate statistics of the best_solution_fitness_history
    min_fitness, max_fitness, mean_fitness, std_fitness = calculate_statistics(best_solution_fitness_history)

    with open(filename+".txt", 'w') as f:
        f.write(f"{obj_func} + {bounds} + {algo}\n")
        f.write("Stats:" + "\n" +
                "Min. fitness : " + str(min_fitness) + "\n" +
                "Max. fitness : " + str(max_fitness) + "\n" +
                "Mean fitness : " + str(mean_fitness) + "\n" +
                "Std. Dev. : " + str(std_fitness)   + "\n" +
                "Elapsed time : " + str(elapsed_time) + " s\n")

    f.close()

    # Calculate sum of corresponding indexes for all the 30 runs and divide by 30 to get avg of that run
    fitness_progress_avg_30_runs = np.sum(fitness_progress_avg_30_runs, axis=0) / 30

    plt.plot(fitness_progress_avg_30_runs, linewidth=0.5)
    plt.title(title + " best fitness convergence", fontsize=20)
    plt.xlabel("Number of iterations", fontsize=10)
    plt.ylabel("Fitness", fontsize=10)
    plt.text(0.65, 0.95, f"Min Fitness: {min_fitness}", transform=plt.gca().transAxes, va='top', fontsize=14)
    plt.text(0.65, 0.9, f"Max Fitness: {max_fitness}", transform=plt.gca().transAxes, va='top', fontsize=14)
    plt.text(0.65, 0.85, f"Mean Fitness: {mean_fitness}", transform=plt.gca().transAxes, va='top', fontsize=14)
    plt.text(0.65, 0.8, f"Std. Dev.: {std_fitness}", transform=plt.gca().transAxes, va='top', fontsize=14)
    plt.savefig(f"./Fitness_Convergence_Graphs/{filename}.png")
    plt.clf()

    return fitness_progress_avg_30_runs


def calculate_statistics(best_solution_fitness_history):
    # Calculate statistics of the best_solution_fitness_history
    min_fitness = min(best_solution_fitness_history)
    max_fitness = max(best_solution_fitness_history)
    mean_fitness = np.mean(best_solution_fitness_history)
    std_fitness = np.std(best_solution_fitness_history)
    return min_fitness, max_fitness, mean_fitness, std_fitness


def plot_fitness_comparison(func1_name, func2_name, obj_fc_and_dimensions, fitness_history1, fitness_history2):
    plt.plot(fitness_history1, label=func1_name, linewidth=0.5)
    plt.plot(fitness_history2, label=func2_name, linewidth=0.5)
    plt.title(f"{func1_name} vs {func2_name} best fitness AVG convergence", fontsize=30)
    plt.xlabel("Number of iterations", fontsize=20)
    plt.ylabel("Fitness", fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig(f"./Fitness_Comparison/{func1_name} " f" {func2_name} " f" {obj_fc_and_dimensions}.png")
    plt.clf()