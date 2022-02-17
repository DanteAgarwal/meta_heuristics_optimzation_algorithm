import math
import random
import time
import matplotlib.pyplot as plt
import numpy as np


def s1(x):
    s1 = 1 / (1 + np.exp(-2 * x))
    return s1


def get_cuckoos(nest, best, lb, ub, n, dim):
    # perform Levy flights
    tempnest = np.zeros((n, dim))
    tempnest = np.array(nest)
    beta = 3 / 2
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (
            math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)

    s = np.zeros(dim)
    for j in range(0, n):
        s = nest[j, :]
        u = np.random.randn(len(s)) * sigma
        v = np.random.randn(len(s))
        step = u / abs(v) ** (1 / beta)

        stepsize = 0.01 * (step * (s - best))

        s = s + stepsize * np.random.randn(len(s))
        tempnest[j, :] = s1(s)
        for i in range(0, dim):
            ss = s1(tempnest[j, i])
            if (random.random() < ss):
                tempnest[j, i] = 1
            else:
                tempnest[j, i] = 0

        while np.sum(tempnest[j, :]) == 0:
            tempnest[j, :] = np.random.randint(2, size=(1, dim))
        # tempnest[j,:]=np.clip(s, lb, ub)
    return tempnest


def get_best_nest(nest, newnest, fitness, n, dim, objf):
    # Evaluating all new solutions
    tempnest = np.zeros((n, dim))
    tempnest = np.copy(nest)
    for j in range(0, n):
        # for j=1:size(nest,1),
        fnew = objf(newnest[j, :])

        if fnew <= fitness[j]:
            fitness[j] = fnew
            tempnest[j, :] = newnest[j, :]

    # Find the current best

    fmin = min(fitness)
    K = np.argmin(fitness)
    bestlocal = tempnest[K, :]

    return fmin, bestlocal, tempnest, fitness


# Replace some nests by constructing new solutions/nests
def empty_nests(nest, pa, n, dim):
    # Discovered or not 
    tempnest = np.zeros((n, dim))

    K = np.random.uniform(0, 1, (n, dim)) > pa
    # K=np.random.randint(2, size=(n,dim))>pa 
    stepsize = random.random() * (nest[np.random.permutation(n), :] - nest[np.random.permutation(n), :])

    tempnest = nest + stepsize * K
    for i in range(0, n):
        for j in range(0, dim):
            if tempnest[i, j] >= .5:
                tempnest[i, j] = 1
            else:
                tempnest[i, j] = 0

    for i in range(0, n):
        while np.sum(tempnest[i, :]) == 0:
            tempnest[i, :] = np.random.randint(2, size=(1, dim))

    #      print(tempnest[j,:])

    return tempnest


##########################################################################


def CS(objf, lb, ub, dim, n, N_IterTotal):
    # lb=-1
    # ub=1
    # n=50
    # N_IterTotal=1000
    # dim=30

    # Discovery rate of alien eggs/solutions
    pa = 0.25

    nd = dim

    #    Lb=[lb]*nd
    #    Ub=[ub]*nd
    convergence1 = []
    convergence2 = []

    # RInitialize nests randomely
    # nest=np.random.rand(n,dim)*(ub-lb)+lb
    nest = np.random.randint(2, size=(n, dim))

    for i in range(0, n):
        while np.sum(nest[i, :]) == 0:
            nest[i, :] = np.random.randint(2, size=(1, dim))

    new_nest = np.zeros((n, dim))
    new_nest = np.copy(nest)

    bestnest = [0] * dim

    fitness = np.zeros(n)
    fitness.fill(float("inf"))

    print("CS is optimizing  \"" + objf.__name__ + "\"")

    timerStart = time.time()

    fmin, bestnest, nest, fitness = get_best_nest(nest, new_nest, fitness, n, dim, objf)
    # Main loop counter
    for iter in range(0, N_IterTotal):
        # Generate new solutions (but keep the current best)

        new_nest = get_cuckoos(nest, bestnest, lb, ub, n, dim)
        # Evaluate new solutions and find best
        fnew, best, nest, fitness = get_best_nest(nest, new_nest, fitness, n, dim, objf)

        new_nest = empty_nests(new_nest, pa, n, dim)

        # Evaluate new solutions and find best
        fnew, best, nest, fitness = get_best_nest(nest, new_nest, fitness, n, dim, objf)

        if fnew < fmin:
            fmin = fnew
            bestnest = best

        featurecount = 0
        for f in range(0, dim):
            if best[f] == 1:
                featurecount = featurecount + 1
        convergence1.append(fmin)
        convergence2.append(featurecount)

        if (iter % 10 == 0):
            print(['At iteration ' + str(iter) + ' the best fitness on trainig is ' + str(
                fmin) + ',the best number of features: ' + str(featurecount)])

    timerEnd = time.time()
    y = np.array(convergence1, dtype=np.longdouble)
    x = np.arange(0, N_IterTotal, dtype=int) + 1
    timerEnd = time.time()
    print('Completed in', (timerEnd - timerStart))
    fire = round((timerEnd - timerStart),2)
    plt.plot(x, y, 'o-')
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.title(
        f"Convergence_curve for CSO for parameter including population "
        f"{n}, \niteration {N_IterTotal},and  max fitness is:{round(min(convergence1),3)}")
    plt.show()

    opts = {"p": bestnest, 'c': round(min(convergence1),3), "ti": fire}

    return opts
