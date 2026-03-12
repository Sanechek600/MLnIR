import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import det, inv

np.random.seed(0)  # чтобы результаты воспроизводились

def bhattacharyya_distance(M0, B0, M1, B1):
    # M0, M1 – (2,), B0, B1 – (2,2)
    B_avg = 0.5 * (B0 + B1)
    term1 = 0.25 * (M1 - M0).T @ inv(B_avg) @ (M1 - M0)
    term2 = 0.5 * np.log(det(B_avg) / np.sqrt(det(B0) * det(B1)))
    return float(term1 + term2)

def mahalanobis_distance(M0, M1, B):
    # B – общая корреляционная матрица
    diff = M1 - M0
    return float(diff.T @ inv(B) @ diff)

def simulate_normal(M, B, N):
    """
    Моделирование N реализаций двумерного нормального вектора X ~ N(M, B)
    """
    B00, B01 = B[0, 0], B[0, 1]
    B11 = B[1, 1]

    a00 = np.sqrt(B00)
    a01 = B01 / a00
    a11 = np.sqrt(B11 - a01**2)

    A = np.array([[a00, a01],
                  [0.0, a11]])
    
    ksi = np.random.randn(N, 2)      # ksi ~ N(0, I)
    X = ksi @ A.T + M                # X = A ksi + M
    return X

def estimate_params(X):
    # X – массив shape (N, 2)
    N = X.shape[0]
    M_hat = X.mean(axis=0)
    # B_hat = (1/N) * sum (x_i - M_hat)(x_i - M_hat)^T
    diff = X - M_hat
    B_hat = (diff.T @ diff) / N
    return M_hat, B_hat

def representative_Sch():
    return np.array([
        [1, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
    ], dtype=np.int8)


def representative_SM():
    return np.array([
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0],
    ], dtype=np.int8)


def generate_binary_samples(rep_2d, N, p):

    rep = np.asarray(rep_2d, dtype=np.int8).reshape(-1, 1)  # (n,1)

    n = rep.shape[0]
    flips = (np.random.random(size=(n, N)) < p).astype(np.int8)
    X = rep ^ flips
    return X  # (n, N)


def plot_binary_vector(vec_2d, title):

    plt.figure()
    plt.imshow(vec_2d, cmap='gray_r', interpolation='nearest')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()