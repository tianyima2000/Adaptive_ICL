import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from random_besov import FastWaveletBatchGenerator


def sample_unit_ball(d, n):
    z = np.random.normal(size=(n, d))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    r = np.random.uniform(size=(n, 1)) ** (1 / d)
    return r * z

def sample_stiefel(d, p):
    if p > d:
        raise ValueError("p must be <= d")

    Z = np.random.normal(size=(d, p))
    Q, R = np.linalg.qr(Z)

    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1
    Q *= signs

    return Q

def sample_projected_vectors(d, p, n):
    if p > d:
        raise ValueError("p must be <= d")
    
    U = sample_stiefel(d, p)
    X = sample_unit_ball(d, n)
    XU = np.matmul(X, U)
    return (XU + 1) / 2



import time

start = time.time()
params_list = []
X_list = []
num_functions = 100
for i in range(num_functions):
    params_list.append({
        'd': 1,
        'alpha': 1,
        'J': 10,
        'seed': np.random.randint(0, 2**30),
        'C0': 1.0
    })

    X_list.append(sample_projected_vectors(10, 1, 100)) 

random_besov_generator = FastWaveletBatchGenerator(wavelet='db6', cascade_level=8)
Y_list = random_besov_generator.generate_batch(X_list, params_list)
random_besov_generator.cleanup()

end = time.time()

print("Elapsed:", end - start, "seconds")