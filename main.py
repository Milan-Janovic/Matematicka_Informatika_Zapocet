import numpy as np
import RS_SA as rs_sa
import matplotlib.pyplot as plt

# Define bounds for the input variables
bounds1_D5 = np.array([[-5, 5]] * 5)
bounds2_D5 = np.array([[-5, 5]] * 5)
bounds3_D5 = np.array([[-500, 500]] * 5)

# Define bounds for the input variables
bounds1_D10 = np.array([[-5, 5]] * 10)
bounds2_D10 = np.array([[-5, 5]] * 10)
bounds3_D10 = np.array([[-500, 500]] * 10)

# Define global minimum for each function
global_FD5 = np.full(5, 0)
global_SD5 = np.full(5, 1)
global_SCH5 = np.full(5, 420.9687)
global_FD10 = np.full(10, 0)
global_SD10 = np.full(10, 1)
global_SCH10 = np.full(10, 420.9687)

# Define max iterations -> ~quadratic time increase
_max_iter_RS = 500
_max_iter_SA = 500

plt.figure(figsize=(24, 12))
# Plot the convergence graph for First Dejong function
RS_FD5_best_fitness_history = rs_sa.plot_convergence(rs_sa.dejong1, bounds1_D5, "First Dejong Function - Random Search Algorithm - D5",
                    "1.RS_FirstDejongConvergenceD5", _max_iter_RS, global_FD5, "random_search")

# Plot the convergence graph for First Dejong function
SA_FD5_best_fitness_history = rs_sa.plot_convergence(rs_sa.dejong1, bounds1_D5, "First Dejong Function - Simulated Annealing - D5",
                    "1.SA_FirstDejongConvergenceD5", _max_iter_SA, global_FD5, "simulated_annealing")

# Plot the comparison
rs_sa.plot_fitness_comparison("Random Search", "Simulated Annealing", "FD5", RS_FD5_best_fitness_history, SA_FD5_best_fitness_history)


# Plot the convergence graph for Second Dejong function
RS_SD5_best_fitness_history = rs_sa.plot_convergence(rs_sa.dejong2, bounds2_D5, "Second Dejong Function - Random Search Algorithm - D5",
                    "2.RS_SecondDejongConvergenceD5", _max_iter_RS, global_SD5, "random_search")

# Plot the convergence graph for Second Dejong function
SA_SD5_best_fitness_history = rs_sa.plot_convergence(rs_sa.dejong2, bounds2_D5, "Second Dejong Function - Simulated Annealing - D5",
                    "2.SA_SecondDejongConvergenceD5", _max_iter_SA, global_SD5, "simulated_annealing")

# Plot the comparison
rs_sa.plot_fitness_comparison("Random Search", "Simulated Annealing", "SD5", RS_SD5_best_fitness_history, SA_SD5_best_fitness_history)


# Plot the convergence graph for Schwefel function
RS_SCH5_best_fitness_history = rs_sa.plot_convergence(rs_sa.schwefel, bounds3_D5, "Schwefel Function - Random Search Algorithm - D5",
                    "3.RS_SchwefelConvergenceD5", _max_iter_RS, global_SCH5, "random_search")

# Plot the convergence graph for Schwefel function
SA_SCH5_best_fitness_history = rs_sa.plot_convergence(rs_sa.schwefel, bounds3_D5, "Schwefel Function - Simulated Annealing - D5",
                    "3.SA_SchwefelConvergenceD5", _max_iter_SA, global_SCH5, "simulated_annealing")

# Plot the comparison
rs_sa.plot_fitness_comparison("Random Search", "Simulated Annealing", "SCH5", RS_SCH5_best_fitness_history, SA_SCH5_best_fitness_history)


RS_FD10_best_fitness_history = rs_sa.plot_convergence(rs_sa.dejong1, bounds1_D10, "First Dejong Function - Random Search Algorithm - D10",
                    "4.RS_FirstDejongConvergenceD10", _max_iter_RS, global_FD10, "random_search")

SA_FD10_best_fitness_history = rs_sa.plot_convergence(rs_sa.dejong1, bounds1_D10, "First Dejong Function - Simulated Annealing - D10",
                    "4.SA_FirstDejongConvergenceD10", _max_iter_SA, global_FD10, "simulated_annealing")

# Plot the comparison
rs_sa.plot_fitness_comparison("Random Search", "Simulated Annealing", "FD10", RS_FD10_best_fitness_history, SA_FD10_best_fitness_history)


# Plot the convergence graph for Second Dejong function
RS_SD10_best_fitness_history = rs_sa.plot_convergence(rs_sa.dejong2, bounds2_D10, "Second Dejong Function - Random Search Algorithm - D10",
                    "5.RS_SecondDejongConvergenceD10", _max_iter_RS, global_SD10, "random_search")

# Plot the convergence graph for Second Dejong function
SA_SD10_best_fitness_history = rs_sa.plot_convergence(rs_sa.dejong2, bounds2_D10, "Second Dejong Function - Simulated Annealing - D10",
                    "5.SA_SecondDejongConvergenceD10", _max_iter_SA, global_SD10, "simulated_annealing")

# Plot the comparison
rs_sa.plot_fitness_comparison("Random Search", "Simulated Annealing", "SD10", RS_SD10_best_fitness_history, SA_SD10_best_fitness_history)


# Plot the convergence graph for Schwefel function
RS_SCH10_best_fitness_history = rs_sa.plot_convergence(rs_sa.schwefel, bounds3_D10, "Schwefel Function - Random Search Algorithm - D10",
                    "6.RS_chwefelConvergenceD10", _max_iter_RS, global_SCH10, "random_search")

# Plot the convergence graph for Schwefel function
SA_SCH10_best_fitness_history = rs_sa.plot_convergence(rs_sa.schwefel, bounds3_D10, "Schwefel Function - Simulated Annealing - D10",
                    "6.SA_chwefelConvergenceD10", _max_iter_SA, global_SCH10, "simulated_annealing")

# Plot the comparison
rs_sa.plot_fitness_comparison("Random Search", "Simulated Annealing", "SCH10", RS_SCH10_best_fitness_history, SA_SCH10_best_fitness_history)








                    

