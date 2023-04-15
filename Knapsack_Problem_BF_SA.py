import itertools
from prettytable import PrettyTable
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar


# Brute force algorithm to solve the knapsack problem
def knapsack_brute_force(capacity, items):
    # Initialize the best value, the best subset and fitness_progress  to 0 and empty, respectively
    best_value = 0
    best_subset = []
    fitness_progress = []

    # Iterate over all possible subsets of items
    for i in range(len(items)):
        for subset in itertools.combinations(items, i + 1):
            # Calculate the total weight and value of the current subset
            subset_weight = sum(item[0] for item in subset)
            subset_value = sum(item[1] for item in subset)

            # Check if the weight is within the capacity of the knapsack
            # and if the value is better than the current best value
            if subset_weight <= capacity and subset_value > best_value:
                # Update the best value and best subset
                best_value = subset_value
                best_subset = subset

            fitness_progress.append(best_value)

    # Convert the best subset to a list of item numbers and return the best value and best subset
    return best_value, [items.index(item) + 1 for item in best_subset], fitness_progress


# Adjusted SA algorithm from part 1. (RS_SA.py)
def knapsack_simulated_annealing(capacity, items, initial_temperature, cooling_rate, alpha=0.8, beta=1.0):
    # Initialize the best value, the best subset and fitness_progress  to 0 and empty, respectively
    current_subset = []
    current_value = 0
    fitness_progress = []

    # Initialize the best state and the best value to the current state and the current value
    best_subset = current_subset
    best_value = current_value

    # Initialize the temperature and the iteration counter
    temperature = initial_temperature
    iteration = 0

    # Run the algorithm until the temperature reaches the temperature stopping point
    # or the MAX_ITERS calculated from maxFES of Brute Force algo -> NOT LIKELY TO HAPPEN
    while temperature > 0.01 and iteration < MAX_ITERS_SA:
        # print(best_subset)
        # print(best_value)
        # Select a random neighbor state by randomly adding or removing an item
        neighbor_subset = current_subset[:]
        # Alpha, Beta -> wieghts (bias) for adding/removing and items from subset
        p_add = min(1, alpha * (len(items) - len(neighbor_subset)) / len(items))
        p_remove = min(1, beta * len(neighbor_subset) / len(items))
        if random.random() < p_add and len(neighbor_subset) < len(items):
            # Add a random item to the subset
            available_items = [item for item in items if item not in neighbor_subset]
            if available_items:
                item_to_add = random.choice(available_items)
                neighbor_subset.append(item_to_add)
        elif len(neighbor_subset) > 0 and random.random() < p_remove:
            # Remove a random item from the subset
            item_to_remove = random.choice(neighbor_subset)
            neighbor_subset.remove(item_to_remove)

        # Calculate the value and the weight of the neighbor state
        neighbor_value = sum(item[1] for item in neighbor_subset)
        neighbor_weight = sum(item[0] for item in neighbor_subset)

        # If the neighbor state is better than the current state, accept it
        if neighbor_weight <= capacity and neighbor_value > current_value:
            current_subset = neighbor_subset
            current_value = neighbor_value

            # If the neighbor state is better than the best state, update the best state
            if current_value > best_value:
                best_subset = current_subset
                best_value = current_value

        # If the neighbor state is worse than the current state, accept it with a probability based on the temperature
        else:
            delta = neighbor_value - current_value
            probability = np.exp(delta / temperature)
            if random.random() < probability:
                current_subset = neighbor_subset
                current_value = neighbor_value

        # Decrease the temperature according to the cooling rate
        temperature *= cooling_rate

        # Increase the iteration counter
        iteration += 1

        fitness_progress.append(best_value)

    # Convert the best subset to a list of item numbers and return the best value and best subset
    return best_value, [items.index(item) + 1 for item in best_subset], fitness_progress


def print_all_items(items):
    with open("Knapsack_Fitness_Graphs/RESULTS.txt", 'a') as f:
        # Print the list of all items
        table = PrettyTable()
        table.field_names = ["Item Number", "Weight", "Value"]
        for i, (weight, value) in enumerate(items):
            table.add_row([i + 1, weight, value])
        f.write("All Items:" + "\n")
        f.write(str(table) + "\n")
        f.write("\n")
        f.close()


def print_solution(items, optimal_items, optimal_value, time_taken, algo):
    with open("Knapsack_Fitness_Graphs/RESULTS.txt", 'a') as f:
        # Create a table of the optimal subset of items
        table = PrettyTable()
        table.field_names = ["Item Number", "Weight", "Value"]
        total_weight = 0
        for i, (weight, value) in enumerate(items):
            if i + 1 in optimal_items:
                table.add_row([i + 1, weight, value])
                total_weight += weight

        if algo == "BF":
            f.write("BRUTE FORCE\n")
        elif algo ==  "SA":
            f.write("SIMULATED ANNEALING\n")
        else:
            f.write("WAT????")
            return

        # Print the optimal value, total weight, and table of the optimal subset of items
        f.write(f"Optimal Value: " + str(optimal_value) + "\n")
        f.write("Optimal Weight:" + str(total_weight) + "\n")
        f.write("Optimal Items:" + "\n")
        f.write(str(table) + "\n")
        f.write("Time taken: " + str(time_taken) + " seconds" + "\n")
        f.write("\n")
        f.close()


def plot_value_convergence_BF(fitness_progress):
    plt.plot(fitness_progress)
    plt.title("Knapsack - Brute Force", fontsize=30)
    plt.xlabel("Number of iterations", fontsize=20)
    plt.ylabel("Value", fontsize=20)
    plt.savefig(f"./Knapsack_Fitness_Graphs/KS_BF.png")
    plt.clf()  # clear the figure to avoid overlapping plots


def plot_fitness_comparison(func1_name, func2_name, fitness_history1, fitness_history2):
    plt.plot(fitness_history1, label=func1_name, linewidth=0.5)
    plt.plot(fitness_history2, label=func2_name, linewidth=0.5)
    plt.title(f"{func1_name} vs {func2_name} best fitness convergence", fontsize=30)
    plt.xlabel("Number of iterations", fontsize=20)
    plt.ylabel("Value", fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig(f"./Knapsack_Fitness_Graphs/{func1_name} " f" {func2_name}.png")
    plt.clf()


def calculate_and_plot_convergence_SA(CAPACITY, items, initial_temperature, cooling_rate, number_of_runs):
    best_value = 0
    best_solution = None
    best_weight = 0
    bar = Bar('Iteration : ', max=30)
    for i in range(number_of_runs):
        bar.next()
        optimal_value_SA_iter, optimal_items_SA_iter, fitness_progress_SA = knapsack_simulated_annealing(CAPACITY, items,
                                                                                                   initial_temperature,
                                                                                                   cooling_rate)
        total_weight_iter = 0
        for i, (weight, value) in enumerate(items):
            if i + 1 in optimal_items_SA_iter:
                total_weight_iter += weight

        if optimal_value_SA_iter > best_value and total_weight_iter < CAPACITY:
            best_value = optimal_value_SA_iter
            best_solution = optimal_items_SA_iter
            best_weight = total_weight_iter


        plt.plot(fitness_progress_SA, linewidth=0.5)

    bar.finish()

    plt.title(f"Knapsack - Simulated Annealing ({number_of_runs} runs)", fontsize=30)
    plt.xlabel("Number of iterations", fontsize=20)
    plt.ylabel("Value", fontsize=20)
    plt.savefig(f"./Knapsack_Fitness_Graphs/KS_SA.png")
    plt.clf()  # clear the figure to avoid overlapping plots

    return best_value, best_solution


# Set plot
plt.figure(figsize=(18, 9))

# Set a number of items to choose from and knapsack capacity
ITEMS_NUMBER = 25
CAPACITY = 200

# Set SA parameters
initial_temperature = 100000
cooling_rate = 0.99

# Calculater maximal number of subsets for BF, to limit SA maxFES later
MAX_ITERS_SA = (2**ITEMS_NUMBER) - 1

# Generate a list of random items and a capacity for the knapsack
items = [(random.randint(1, 50), random.randint(1, 50)) for _ in range(ITEMS_NUMBER)]

# Print all items
print_all_items(items)


"""

BF

"""

# Start timer
start_time = time.time()

# Solve the knapsack problem using the brute force algorithm
optimal_value_BF, optimal_items_BF, fitness_progress_BF = knapsack_brute_force(CAPACITY, items)

# End timer and calculate the time taken to solve the problem
end_time = time.time()
time_taken = end_time - start_time

print("\nBRUTE FORCE")

# Print solution
print_solution(items, optimal_items_BF, optimal_value_BF, time_taken, "BF")

# Plot
plot_value_convergence_BF(fitness_progress_BF)

"""

SA

"""

# Solve the knapsack problem using the simulated annealing algorithm
# Start timer
start_time = time.time()

# Calculate and Plot
optimal_value_SA, optimal_items_SA = calculate_and_plot_convergence_SA(CAPACITY, items, initial_temperature,
                                                                       cooling_rate, number_of_runs=1)

# End timer and calculate the time taken to solve the problem
end_time = time.time()
time_taken = end_time - start_time

print("\nSIMULATED ANNEALING")

# Print solution
print_solution(items, optimal_items_SA, optimal_value_SA, time_taken, "SA")


"""

COMPARE -> Not sure if should be done as iterations don't match ???

"""

# Plot comparison
# plot_fitness_comparison("BF", "SA", fitness_progress_BF, fitness_progress_SA)

