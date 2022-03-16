import numpy as np

"""
From paper by Maekawa et al.

https://gem-ecmlpkdd.github.io/archive/2019/papers/GEM2019_paper_15.pdf
https://github.com/seijimaekawa/acMark

Parameters description:
    - alpha: parameters for balancing inter-edges and intra-edges
    - beta: parameters of separability for attribute cluster proportions
    - sigma: deviations of normal distribution
    - omega : parameters of random attributes
    
"""

# =========VARYING PARAMS=========
alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
beta = [0.1]
sigma = [0.1]
omega = [0.1]

RUN = list(range(1, 6))

# ========FIXED PARAMS=========
n = 5000
m = 2 ** (10 + 5)
max_deg = 500
k = 6
d = 8
phi_c = 1

M = np.array([
    [0.6, 0.08, 0.08, 0.08, 0.08, 0.08],
    [0.08, 0.6, 0.08, 0.08, 0.08, 0.08],
    [0.08, 0.08, 0.6, 0.08, 0.08, 0.08],
    [0.08, 0.08, 0.08, 0.6, 0.08, 0.08],
    [0.08, 0.08, 0.08, 0.08, 0.6, 0.08],
    [0.08, 0.08, 0.08, 0.08, 0.08, 0.6]
])

D = np.array([
    [0.2, 0.1, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.2, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.25, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.25, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.3, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.3]
])
