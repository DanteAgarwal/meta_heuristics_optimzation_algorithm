import math
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool, cpu_count


def s1(x):
    return 1 / (1 + np.exp(-2 * x))


def evaluate_population_mp(objf, population):
    with Pool(processes=cpu_count()) as pool:
        return np.array(pool.map(objf, population))


def get_cuckoos(nest, best, n, dim):
    beta = 3 / 2
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)

    new_nest = np.copy(nest)
    for j in range(n):
        u = np.random.randn(dim) * sigma
        v = np.random.randn(dim)
        step = u / (np.abs(v) ** (1 / beta))
        stepsize = 0.01 * step * (nest[j] - best)
        s = nest[j] + stepsize * np.random.randn(dim)
        prob = s1(s)
        new_nest[j] = np.where(np.random.rand(dim) < prob, 1, 0)
        while np.sum(new_nest[j]) == 0:
            new_nest[j] = np.random.randint(2, size=dim)
    return new_nest


def get_best_nest(nest, new_nest, fitness, objf):
    n = len(nest)
    dim = nest.shape[1]
    new_fitness = evaluate_population_mp(objf, new_nest)
    improved = new_fitness <= fitness
    fitness = np.where(improved, new_fitness, fitness)
    nest[improved] = new_nest[improved]
    best_idx = np.argmin(fitness)
    return fitness[best_idx], nest[best_idx].copy(), nest, fitness


def empty_nests(nest, pa, n, dim):
    K = np.random.rand(n, dim) > pa
    stepsize = np.random.rand() * (nest[np.random.permutation(n)] - nest[np.random.permutation(n)])
    new_nest = nest + stepsize * K
    new_nest = np.where(new_nest >= 0.5, 1, 0)
    for i in range(n):
        if np.sum(new_nest[i]) == 0:
            new_nest[i] = np.random.randint(2, size=dim)
    return new_nest


def CS(objf, lb, ub, dim, n, N_IterTotal):
    pa = 0.25
    nest = np.random.randint(2, size=(n, dim))
    for i in range(n):
        while np.sum(nest[i]) == 0:
            nest[i] = np.random.randint(2, size=dim)

    fitness = np.full(n, float("inf"))
    best_fitness, best_nest, nest, fitness = get_best_nest(nest, nest.copy(), fitness, objf)

    convergence = []
    feature_counts = []

    print(f'CS is optimizing "{objf.__name__}"')
    timer_start = time.time()

    for iter in range(N_IterTotal):
        new_nest = get_cuckoos(nest, best_nest, n, dim)
        best_fitness, best_nest, nest, fitness = get_best_nest(nest, new_nest, fitness, objf)

        new_nest = empty_nests(nest, pa, n, dim)
        best_fitness, best_nest, nest, fitness = get_best_nest(nest, new_nest, fitness, objf)

        convergence.append(best_fitness)
        feature_counts.append(np.sum(best_nest))

        if iter % 10 == 0:
            print(f"Iter {iter}: Fitness = {best_fitness:.4f}, Features = {int(np.sum(best_nest))}")

    elapsed_time = round(time.time() - timer_start, 2)
    print(f"Completed in {elapsed_time}s")

    x = np.arange(1, N_IterTotal + 1)
    y = np.array(convergence)
    plt.plot(x, y, 'o-')
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.title(
        f"CSO Convergence\nPop: {n}, Iters: {N_IterTotal}, "
        f"Best Fitness: {round(best_fitness, 3)}"
    )
    plt.show()

    return {"p": best_nest, "c": round(best_fitness, 3), "ti": elapsed_time}
