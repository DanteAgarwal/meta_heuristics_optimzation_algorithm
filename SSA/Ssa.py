import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count


def evaluate_population_mp(objf, population):
    """Parallel evaluation of fitness for a population using multiprocessing."""
    with Pool(processes=cpu_count()) as pool:
        fitness = pool.map(objf, population)
    return np.array(fitness)


def SSA(objf, lb, ub, dim, N, Max_iteration):
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    lb = np.array(lb)
    ub = np.array(ub)

    SalpPositions = np.random.uniform(lb, ub, (N, dim))
    SalpFitness = np.full(N, np.inf)
    Convergence_curve = np.zeros(Max_iteration)

    print(f'SSA is optimizing "{objf.__name__}"')

    timerStart = time.time()

    # Initial evaluation
    SalpFitness = evaluate_population_mp(objf, SalpPositions)
    best_idx = np.argmin(SalpFitness)
    FoodPosition = SalpPositions[best_idx].copy()
    FoodFitness = SalpFitness[best_idx]

    for Iteration in range(Max_iteration):
        c1 = 2 * math.exp(-((4 * Iteration / Max_iteration) ** 2))

        for i in range(N):
            if i < N / 2:
                c2 = np.random.rand(dim)
                c3 = np.random.rand(dim)
                direction = np.where(c3 < 0.5, 1, -1)
                move = direction * c1 * ((ub - lb) * c2 + lb)
                SalpPositions[i] = FoodPosition + move
            else:
                if i > 0:
                    SalpPositions[i] = (SalpPositions[i - 1] + SalpPositions[i]) / 2

        SalpPositions = np.clip(SalpPositions, lb, ub)

        # Parallel fitness evaluation
        SalpFitness = evaluate_population_mp(objf, SalpPositions)

        min_idx = np.argmin(SalpFitness)
        if SalpFitness[min_idx] < FoodFitness:
            FoodFitness = SalpFitness[min_idx]
            FoodPosition = SalpPositions[min_idx].copy()

        Convergence_curve[Iteration] = FoodFitness
        print(f"Iteration {Iteration+1}: Best Fitness = {FoodFitness:.6f}")

    total_time = round(time.time() - timerStart, 2)
    print(f"Completed in {total_time} seconds")

    # Plotting
    plt.plot(np.arange(1, Max_iteration + 1), Convergence_curve, 'o-')
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.title(f"SSA Convergence\nBest={round(FoodFitness, 3)} in {total_time}s")
    plt.show()

    return {"p": FoodPosition, "c": round(FoodFitness, 3), "ti": total_time}
