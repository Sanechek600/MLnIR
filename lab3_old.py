import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.linalg import inv, det
from utility import *

# ================== Вспомогательные функции ==================

def bayes_classifier_normal_equal_B(x, M0, M1, B, P0=0.5, P1=0.5):
    """Байесовский классификатор для равных корреляционных матриц.
       Возвращает 0, если класс 0, иначе 1."""
    B_inv = inv(B)
    d0 = M0.T @ B_inv @ x - 0.5 * M0.T @ B_inv @ M0 + np.log(P0)
    d1 = M1.T @ B_inv @ x - 0.5 * M1.T @ B_inv @ M1 + np.log(P1)
    return 0 if d0 > d1 else 1

def bayes_classifier_normal_unequal_B(x, M0, B0, M1, B1, P0=0.5, P1=0.5):
    """Байесовский классификатор для разных корреляционных матриц."""
    def disc(x, M, B, P):
        B_inv = inv(B)
        return -0.5 * (x - M).T @ B_inv @ (x - M) - 0.5 * np.log(det(B)) + np.log(P)
    d0 = disc(x, M0, B0, P0)
    d1 = disc(x, M1, B1, P1)
    return 0 if d0 > d1 else 1

def get_errors(X0, X1, classifier_func):
    """Экспериментальная оценка вероятностей ошибок."""
    err0 = sum(1 for x in X0 if classifier_func(x) != 0) / len(X0)
    err1 = sum(1 for x in X1 if classifier_func(x) != 1) / len(X1)
    return err0, err1

def compute_fisher_params(M0, B0, M1, B1):
    """
    Вычисляет параметры линейного классификатора Фишера
    по формулам (3.15) и (3.16) для неравных ковариаций.
    Возвращает W, wN.
    """
    B_avg = 0.5 * (B0 + B1)
    B_avg_inv = inv(B_avg)
    W = B_avg_inv @ (M1 - M0)
    sigma0_2 = W.T @ B0 @ W
    sigma1_2 = W.T @ B1 @ W
    wN = - (1.0 / (sigma0_2 + sigma1_2)) * (M1 - M0).T @ B_avg_inv @ (sigma1_2 * M0 + sigma0_2 * M1)
    return W, wN

def compute_fisher_params_equal_B(M0, M1, B):
    """Классификатор Фишера для равных ковариаций – формулы (3.15'), (3.16')."""
    B_inv = inv(B)
    W = B_inv @ (M1 - M0)
    wN = -0.5 * (M1 - M0).T @ B_inv @ (M0 + M1)
    return W, wN

def compute_mse_params(M0, B0, M1, B1, P0=0.5, P1=0.5):
    """
    Вычисляет параметры линейного классификатора, минимизирующего СКО
    на генеральной совокупности. Используется расширенный вектор Z = (X, 1)
    и целевое значение t = +1 для класса 1, t = -1 для класса 0.
    Возвращает W (вектор весов для исходных признаков) и wN (порог).
    """
    E0 = np.concatenate([M0, [1.0]])
    E1 = np.concatenate([M1, [1.0]])
    R0 = np.zeros((3, 3))
    R0[:2, :2] = B0 + np.outer(M0, M0)
    R0[:2, 2] = M0
    R0[2, :2] = M0
    R0[2, 2] = 1.0
    R1 = np.zeros((3, 3))
    R1[:2, :2] = B1 + np.outer(M1, M1)
    R1[:2, 2] = M1
    R1[2, :2] = M1
    R1[2, 2] = 1.0
    M_z_t = P0 * (-1.0) * E0 + P1 * (+1.0) * E1
    R_zz = P0 * R0 + P1 * R1
    W_ext = inv(R_zz) @ M_z_t
    W = W_ext[:2]
    wN = W_ext[2]
    return W, wN

def linear_classifier(x, W, wN):
    """Линейный классификатор: возвращает 0 если W^T x + wN > 0, иначе 1."""
    val = W @ x + wN
    return 0 if val > 0 else 1

def plot_decision_boundary(X0, X1, classifier_func, title, filename, xlim=None, ylim=None, extra_boundaries=None):
    """Визуализация разделяющей границы."""
    plt.figure(figsize=(8, 6))
    plt.scatter(X0[:, 0], X0[:, 1], s=10, alpha=0.5, label='Класс 0')
    plt.scatter(X1[:, 0], X1[:, 1], s=10, alpha=0.5, label='Класс 1')
    
    if xlim is None:
        xlim = plt.xlim()
    if ylim is None:
        ylim = plt.ylim()
    
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 200),
                         np.linspace(ylim[0], ylim[1], 200))
    
    # Вычисляем Z для каждой точки сетки
    Z = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            Z[i, j] = classifier_func(np.array([xx[i, j], yy[i, j]]))
    
    plt.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, label='Граница')
    
    if extra_boundaries:
        for W, wN, label, color, ls in extra_boundaries:
            if np.abs(W[1]) > 1e-6:
                y_vals = -(W[0] * xx + wN) / W[1]
                plt.plot(xx[0, :], y_vals, color=color, linestyle=ls, linewidth=2, label=label)
            else:
                x_vals = -wN / W[0]
                plt.axvline(x_vals, color=color, linestyle=ls, linewidth=2, label=label)
    
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()

# ================== Задание 1: Классификатор Фишера ==================
def task1_fisher():
    print("Задание 1: Линейный классификатор, максимизирующий критерий Фишера")
    # Загрузка данных для равных ковариаций
    X0_eq = np.load("normal_eq_1.npy")
    X1_eq = np.load("normal_eq_2.npy")
    M0 = np.array([0.0, 1.0])
    M1 = np.array([1.0, -1.0])
    B_eq = np.array([[1.0, 0.4], [0.4, 1.0]])
    # Для неравных ковариаций (используем два класса из трёх)
    X0_neq = np.load("normal_3_1.npy")
    X1_neq = np.load("normal_3_2.npy")
    B0 = np.array([[1.0, 0.1], [0.1, 1.0]])
    B1 = np.array([[1.0, 0.4], [0.4, 1.0]])
    
    # === Равные ковариации ===
    W_f, wN_f = compute_fisher_params_equal_B(M0, M1, B_eq)
    classifier_fisher = lambda x: linear_classifier(x, W_f, wN_f)
    classifier_bayes = lambda x: bayes_classifier_normal_equal_B(x, M0, M1, B_eq)
    p0_f, p1_f = get_errors(X0_eq, X1_eq, classifier_fisher)
    p0_b, p1_b = get_errors(X0_eq, X1_eq, classifier_bayes)
    print("Равные ковариации:")
    print(f"  Фишер:     p0={p0_f:.4f}, p1={p1_f:.4f}")
    print(f"  Байес:     p0={p0_b:.4f}, p1={p1_b:.4f}")
    # Визуализация
    plot_decision_boundary(X0_eq, X1_eq, classifier_fisher,
        "Классификатор Фишера (равные ковариации)", "fisher_eq.png",
        extra_boundaries=[(W_f, wN_f, "Фишер", "red", "-")])
    
    # === Неравные ковариации ===
    W_f, wN_f = compute_fisher_params(M0, B0, M1, B1)
    classifier_fisher = lambda x: linear_classifier(x, W_f, wN_f)
    classifier_bayes = lambda x: bayes_classifier_normal_unequal_B(x, M0, B0, M1, B1)
    p0_f, p1_f = get_errors(X0_neq, X1_neq, classifier_fisher)
    p0_b, p1_b = get_errors(X0_neq, X1_neq, classifier_bayes)
    print("Неравные ковариации:")
    print(f"  Фишер:     p0={p0_f:.4f}, p1={p1_f:.4f}")
    print(f"  Байес:     p0={p0_b:.4f}, p1={p1_b:.4f}")
    plot_decision_boundary(X0_neq, X1_neq, classifier_fisher,
        "Классификатор Фишера (неравные ковариации)", "fisher_neq.png",
        extra_boundaries=[(W_f, wN_f, "Фишер", "red", "-")])

# ================== Задание 2: Классификатор, минимизирующий СКО ==================
def task2_mse():
    print("Задание 2: Линейный классификатор, минимизирующий среднеквадратичную ошибку")
    X0_eq = np.load("normal_eq_1.npy")
    X1_eq = np.load("normal_eq_2.npy")
    X0_neq = np.load("normal_3_1.npy")
    X1_neq = np.load("normal_3_2.npy")
    M0 = np.array([0.0, 1.0])
    M1 = np.array([1.0, -1.0])
    B_eq = np.array([[1.0, 0.4], [0.4, 1.0]])
    B0 = np.array([[1.0, 0.1], [0.1, 1.0]])
    B1 = np.array([[1.0, 0.4], [0.4, 1.0]])
    
    # === Равные ковариации ===
    W_mse, wN_mse = compute_mse_params(M0, B_eq, M1, B_eq, P0=0.5, P1=0.5)
    classifier_mse = lambda x: linear_classifier(x, W_mse, wN_mse)
    W_f, wN_f = compute_fisher_params_equal_B(M0, M1, B_eq)
    classifier_fisher = lambda x: linear_classifier(x, W_f, wN_f)
    classifier_bayes = lambda x: bayes_classifier_normal_equal_B(x, M0, M1, B_eq)
    p0_mse, p1_mse = get_errors(X0_eq, X1_eq, classifier_mse)
    p0_f, p1_f = get_errors(X0_eq, X1_eq, classifier_fisher)
    p0_b, p1_b = get_errors(X0_eq, X1_eq, classifier_bayes)
    print("Равные ковариации:")
    print(f"  MSE:       p0={p0_mse:.4f}, p1={p1_mse:.4f}")
    print(f"  Фишер:     p0={p0_f:.4f}, p1={p1_f:.4f}")
    print(f"  Байес:     p0={p0_b:.4f}, p1={p1_b:.4f}")
    plot_decision_boundary(X0_eq, X1_eq, classifier_mse,
        "MSE-классификатор (равные ковариации)", "mse_eq.png",
        extra_boundaries=[(W_mse, wN_mse, "MSE", "blue", "-"),
                          (W_f, wN_f, "Фишер", "green", "--")])
    
    # === Неравные ковариации ===
    W_mse, wN_mse = compute_mse_params(M0, B0, M1, B1, P0=0.5, P1=0.5)
    classifier_mse = lambda x: linear_classifier(x, W_mse, wN_mse)
    W_f, wN_f = compute_fisher_params(M0, B0, M1, B1)
    classifier_fisher = lambda x: linear_classifier(x, W_f, wN_f)
    classifier_bayes = lambda x: bayes_classifier_normal_unequal_B(x, M0, B0, M1, B1)
    p0_mse, p1_mse = get_errors(X0_neq, X1_neq, classifier_mse)
    p0_f, p1_f = get_errors(X0_neq, X1_neq, classifier_fisher)
    p0_b, p1_b = get_errors(X0_neq, X1_neq, classifier_bayes)
    print("Неравные ковариации:")
    print(f"  MSE:       p0={p0_mse:.4f}, p1={p1_mse:.4f}")
    print(f"  Фишер:     p0={p0_f:.4f}, p1={p1_f:.4f}")
    print(f"  Байес:     p0={p0_b:.4f}, p1={p1_b:.4f}")
    plot_decision_boundary(X0_neq, X1_neq, classifier_mse,
        "MSE-классификатор (неравные ковариации)", "mse_neq.png",
        extra_boundaries=[(W_mse, wN_mse, "MSE", "blue", "-"),
                          (W_f, wN_f, "Фишер", "green", "--")])

# ================== Задание 3: Стохастическая аппроксимация (Роббинс-Монро) ==================
def robbins_monro_akp(X0, X1, alphas, W_init=None):
    """
    АКП-алгоритм (абсолютная ошибка).
    X0, X1 – обучающие выборки (матрицы N x 2).
    alphas – последовательность коэффициентов (итерация -> alpha).
    Возвращает историю векторов W (включая порог) и ошибки.
    """
    N0 = len(X0)
    N1 = len(X1)
    X_ext = np.vstack([np.hstack([X0, np.ones((N0, 1))]),
                       np.hstack([X1, np.ones((N1, 1))])])
    r = np.concatenate([-np.ones(N0), np.ones(N1)])
    idx = np.random.permutation(len(r))
    X_ext = X_ext[idx]
    r = r[idx]
    
    if W_init is None:
        W = np.zeros(3)
    else:
        W = W_init.copy()
    history_W = [W.copy()]
    errors = []
    max_epochs = len(alphas) // len(r) + 1
    epoch = 0
    for k, alpha in enumerate(alphas):
        i = k % len(r)
        xk = X_ext[i]
        rk = r[i]
        if rk >= W @ xk:
            W = W + alpha * xk
        else:
            W = W - alpha * xk
        history_W.append(W.copy())
        if (k+1) % len(r) == 0:
            epoch += 1
            pred = (X_ext @ W) > 0
            err = np.mean(pred != (r == 1))
            errors.append(err)
            if epoch >= 200:
                break
    return np.array(history_W), np.array(errors)

def robbins_monro_nsko(X0, X1, alphas, W_init=None):
    """
    НСКО-алгоритм (наименьшая СКО).
    """
    N0 = len(X0)
    N1 = len(X1)
    X_ext = np.vstack([np.hstack([X0, np.ones((N0, 1))]),
                       np.hstack([X1, np.ones((N1, 1))])])
    r = np.concatenate([-np.ones(N0), np.ones(N1)])
    idx = np.random.permutation(len(r))
    X_ext = X_ext[idx]
    r = r[idx]
    
    if W_init is None:
        W = np.zeros(3)
    else:
        W = W_init.copy()
    history_W = [W.copy()]
    errors = []
    max_epochs = len(alphas) // len(r) + 1
    epoch = 0
    for k, alpha in enumerate(alphas):
        i = k % len(r)
        xk = X_ext[i]
        rk = r[i]
        err_val = rk - W @ xk
        W = W + alpha * xk * err_val
        history_W.append(W.copy())
        if (k+1) % len(r) == 0:
            epoch += 1
            pred = (X_ext @ W) > 0
            err = np.mean(pred != (r == 1))
            errors.append(err)
            if epoch >= 200:
                break
    return np.array(history_W), np.array(errors)

def task3_robbins_monro():
    print("Задание 3: Линейный классификатор на основе процедуры Роббинса-Монро")
    X0_eq = np.load("normal_eq_1.npy")
    X1_eq = np.load("normal_eq_2.npy")
    X0_neq = np.load("normal_3_1.npy")
    X1_neq = np.load("normal_3_2.npy")
    
    beta = 0.6
    max_iter = 5000
    alphas = np.array([1.0 / (i+1)**beta for i in range(max_iter)])
    
    def investigate(data_pair, title_tag):
        X0, X1 = data_pair
        init_zero = np.zeros(3)
        init_rand = np.random.randn(3) * 0.1
        W_akp_zero, err_akp_zero = robbins_monro_akp(X0, X1, alphas, init_zero)
        W_akp_rand, err_akp_rand = robbins_monro_akp(X0, X1, alphas, init_rand)
        W_nsko_zero, err_nsko_zero = robbins_monro_nsko(X0, X1, alphas, init_zero)
        W_nsko_rand, err_nsko_rand = robbins_monro_nsko(X0, X1, alphas, init_rand)
        
        plt.figure(figsize=(10, 6))
        plt.plot(err_akp_zero, label='АКП, W0=0')
        plt.plot(err_akp_rand, label='АКП, W0=random')
        plt.plot(err_nsko_zero, label='НСКО, W0=0')
        plt.plot(err_nsko_rand, label='НСКО, W0=random')
        plt.xlabel('Эпоха')
        plt.ylabel('Вероятность ошибки')
        plt.title(f'Сходимость алгоритмов ({title_tag})')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'robbins_monro_{title_tag}.png')
        plt.close()
        
        final_W_akp = W_akp_zero[-1]
        final_W_nsko = W_nsko_zero[-1]
        class_akp = lambda x: linear_classifier(x, final_W_akp[:2], final_W_akp[2])
        class_nsko = lambda x: linear_classifier(x, final_W_nsko[:2], final_W_nsko[2])
        p0_akp, p1_akp = get_errors(X0, X1, class_akp)
        p0_nsko, p1_nsko = get_errors(X0, X1, class_nsko)
        print(f"{title_tag}:")
        print(f"  АКП:   p0={p0_akp:.4f}, p1={p1_akp:.4f}")
        print(f"  НСКО:  p0={p0_nsko:.4f}, p1={p1_nsko:.4f}")
        
        plot_decision_boundary(X0, X1, class_nsko,
            f"НСКО-классификатор ({title_tag})", f"nsko_final_{title_tag}.png")
    
    print("Исследование для равных ковариаций:")
    investigate((X0_eq, X1_eq), "equal_cov")
    print("Исследование для неравных ковариаций:")
    investigate((X0_neq, X1_neq), "unequal_cov")

# ================== Главная функция ==================
if __name__ == "__main__":
    task1_fisher()
    task2_mse()
    task3_robbins_monro()