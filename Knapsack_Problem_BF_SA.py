import itertools
from prettytable import PrettyTable
import time
import random
import numpy as np

# Brute force algorithm to solve the knapsack problem
def knapsack_brute_force(capacity, items):
    # Initialize the best value and best subset to 0 and empty, respectively
    best_value = 0
    best_subset = []

    # Iterate over all possible subsets of items
    for i in range(len(items)):
        for subset in itertools.combinations(items, i + 1):
            # Calculate the total weight and value of the current subset
            subset_weight = sum(item[0] for item in subset)
            subset_value = sum(item[1] for item in subset)

            # Check if the weight is within the capacity of the knapsack and if the value is better than the current best value
            if subset_weight <= capacity and subset_value > best_value:
                # Update the best value and best subset
                best_value = subset_value
                best_subset = subset

    # Convert the best subset to a list of item numbers and return the best value and best subset
    return best_value, [items.index(item) + 1 for item in best_subset]

def knapsack_simulated_annealing(capacity, items, initial_temperature, cooling_rate):
    # Initialize the current state as an empty subset and the current value as 0
    current_subset = []
    current_value = 0

    # Initialize the best state and the best value to the current state and the current value
    best_subset = current_subset
    best_value = current_value

    # Initialize the temperature and the iteration counter
    temperature = initial_temperature
    iteration = 0

    # Run the algorithm until the temperature reaches aprox. 0
    while temperature > 0.001:
        # Select a random neighbor state by randomly adding or removing an item
        neighbor_subset = current_subset[:]
        if random.random() < 1/len(items) and len(neighbor_subset) < len(items):
            # Add a random item to the subset
            available_items = [item for item in items if item not in neighbor_subset]
            if available_items:
                item_to_add = random.choice(available_items)
                neighbor_subset.append(item_to_add)
        elif len(neighbor_subset) > 0 and random.random() < (1 - 1/len(items)):
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

    # Convert the best subset to a list of item numbers and return the best value and best subset
    return best_value, [items.index(item) + 1 for item in best_subset]

# Generate a list of random items and a capacity for the knapsack
items = [(random.randint(1, 50), random.randint(1, 50)) for _ in range(20)]
capacity = 200

# Start timer
start_time = time.time()

# Print the list of all items
table = PrettyTable()
table.field_names = ["Item Number", "Weight", "Value"]
for i, (weight, value) in enumerate(items):
    table.add_row([i+1, weight, value])
print("All Items:")
print(table)

# Solve the knapsack problem using the brute force algorithm
optimal_value, optimal_items = knapsack_brute_force(capacity, items)

# Create a table of the optimal subset of items
table = PrettyTable()
table.field_names = ["Item Number", "Weight", "Value"]
total_weight = 0
for i, (weight, value) in enumerate(items):
    if i+1 in optimal_items:
        table.add_row([i+1, weight, value])
        total_weight += weight

# Print the optimal value, total weight, and table of the optimal subset of items
print("Optimal Value:", optimal_value)
print("Optimal Weight:", total_weight)
print("Optimal Items:")
print(table)

# End timer and print the time taken to solve the problem
end_time = time.time()
print("Time taken:", end_time - start_time, "seconds")

"""

SA

"""

# Start timer
start_time = time.time()

# Solve the knapsack problem using the simulated annealing algorithm
initial_temperature = 100
cooling_rate = 0.99
optimal_value, optimal_items = knapsack_simulated_annealing(capacity, items, initial_temperature, cooling_rate)

# Create a table of the optimal subset of items
table = PrettyTable()
table.field_names = ["Item Number", "Weight", "Value"]
total_weight = 0
for i, (weight, value) in enumerate(items):
    if i+1 in optimal_items:
        table.add_row([i+1, weight, value])
        total_weight += weight

# Print the optimal value, total weight, and table of the optimal subset of items
print("Optimal Value:", optimal_value)
print("Optimal Weight:", total_weight)
print("Optimal Items:")
print(table)

# End timer and print the time taken to solve the problem
end_time = time.time()
print("Time taken:", end_time - start_time, "seconds")
