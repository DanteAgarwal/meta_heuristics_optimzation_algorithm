import numpy as np
import random
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count


def evaluate_population_mp(objf, population):
    """Parallel evaluation of objective function using multiprocessing."""
    with Pool(processes=cpu_count()) as pool:
        fitness = pool.map(objf, population)
    return np.array(fitness)


def GWO(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    lb = np.array(lb)
    ub = np.array(ub)

    # Initialize alpha, beta, and delta
    Alpha_pos = np.zeros(dim)
    Alpha_score = float("inf")

    Beta_pos = np.zeros(dim)
    Beta_score = float("inf")

    Delta_pos = np.zeros(dim)
    Delta_score = float("inf")

    # Initialize positions
    Positions = np.random.uniform(lb, ub, (SearchAgents_no, dim))
    Convergence_curve = np.zeros(Max_iter)

    print(f'GWO is optimizing "{objf.__name__}"')
    timerStart = time.time()

    for l in range(Max_iter):
        # Boundary check
        Positions = np.clip(Positions, lb, ub)

        # Evaluate fitness in parallel
        fitness_vals = evaluate_population_mp(objf, Positions)

        # Rank and update alpha, beta, delta
        for i in range(SearchAgents_no):
            fitness = fitness_vals[i]
            if fitness < Alpha_score:
                Alpha_score, Alpha_pos = fitness, Positions[i].copy()
            elif fitness < Beta_score:
                Beta_score, Beta_pos = fitness, Positions[i].copy()
            elif fitness < Delta_score:
                Delta_score, Delta_pos = fitness, Positions[i].copy()

        a = 2 - l * (2 / Max_iter)  # Linearly decreases from 2 to 0

        # Update position
        for i in range(SearchAgents_no):
            for j in range(dim):
                # Alpha
                r1, r2 = random.random(), random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                X1 = Alpha_pos[j] - A1 * D_alpha

                # Beta
                r1, r2 = random.random(), random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                X2 = Beta_pos[j] - A2 * D_beta

                # Delta
                r1, r2 = random.random(), random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                X3 = Delta_pos[j] - A3 * D_delta

                # Average the positions
                Positions[i, j] = (X1 + X2 + X3) / 3

        Convergence_curve[l] = Alpha_score

        print(f"Iteration {l+1}: Best Fitness = {Alpha_score:.6f}")

    fire = round(time.time() - timerStart, 2)
    print("Completed in", fire, "seconds")

    # Plot
    x = np.arange(1, Max_iter + 1)
    plt.plot(x, Convergence_curve, 'o-')
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.title(
        f"Convergence Curve - GWO\nPopulation: {SearchAgents_no}, Iterations: {Max_iter}, "
        f"Best Fitness: {round(Alpha_score, 3)}"
    )
    plt.show()

    return {"p": Alpha_pos, "c": round(Alpha_score, 3), "ti": fire}
