import numpy as np
import matplotlib.pyplot as plt
import random
import time
from multiprocessing import Pool, cpu_count


def evaluate_population(objf, population):
    with Pool(processes=cpu_count()) as pool:
        return np.array(pool.map(objf, population))


def TLBO(objf, lb, ub, dim, pop_size, iters):
    # Initialize population within bounds
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = evaluate_population(objf, pop)

    # Store best solution
    best_idx = np.argmin(fitness)
    best_pos = pop[best_idx].copy()
    best_score = fitness[best_idx]
    convergence = []

    print(f"TLBO is optimizing: '{objf.__name__}'")

    start_time = time.time()

    for t in range(iters):
        mean = np.mean(pop, axis=0)
        teacher_idx = np.argmin(fitness)
        teacher = pop[teacher_idx]

        # Teacher Phase
        for i in range(pop_size):
            TF = random.choice([1, 2])
            diff = random.random() * (teacher - TF * mean)
            new_sol = pop[i] + diff
            new_sol = np.clip(new_sol, lb, ub)
            new_fit = objf(new_sol)

            if new_fit < fitness[i]:
                pop[i] = new_sol
                fitness[i] = new_fit

        # Learner Phase
        for i in range(pop_size):
            j = random.choice([x for x in range(pop_size) if x != i])
            if fitness[i] < fitness[j]:
                diff = random.random() * (pop[i] - pop[j])
            else:
                diff = random.random() * (pop[j] - pop[i])
            new_sol = pop[i] + diff
            new_sol = np.clip(new_sol, lb, ub)
            new_fit = objf(new_sol)

            if new_fit < fitness[i]:
                pop[i] = new_sol
                fitness[i] = new_fit

        # Update global best
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_score:
            best_score = fitness[best_idx]
            best_pos = pop[best_idx].copy()

        convergence.append(best_score)

        if t % 10 == 0 or t == iters - 1:
            print(f"Iteration {t+1}: Best Fitness = {best_score:.5f}")

    elapsed_time = round(time.time() - start_time, 2)
    print(f"Optimization completed in {elapsed_time}s")

    # Plot convergence
    plt.plot(np.arange(1, iters + 1), convergence, 'o-')
    plt.xlabel('Iterations')
    plt.ylabel('Best Fitness')
    plt.title(f'TLBO Convergence Curve\nBest: {round(best_score, 4)}')
    plt.grid()
    plt.show()

    return {'p': best_pos, 'c': round(best_score, 4), 'ti': elapsed_time}
