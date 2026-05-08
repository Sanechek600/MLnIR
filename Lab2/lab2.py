import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from numpy.linalg import det, inv
from utility import *

# ================== Вспомогательные функции ==================

def bayes_normal_equal_B(x, M0, M1, B, P0=0.5, P1=0.5):
    """
    Дискриминантная функция для нормального распределения с равными матрицами ковариации.
    """
    B_inv = inv(B)
    d0 = M0.T @ B_inv @ x - 0.5 * M0.T @ B_inv @ M0 + np.log(P0)
    d1 = M1.T @ B_inv @ x - 0.5 * M1.T @ B_inv @ M1 + np.log(P1)
    return d0, d1

def bayes_normal_unequal_B(x, M, B, P):
    """
    Дискриминантная функция для нормального распределения с разными матрицами ковариации.
    """
    B_inv = inv(B)
    term1 = -0.5 * (x - M).T @ B_inv @ (x - M)
    term2 = -0.5 * np.log(det(B))
    term3 = np.log(P)
    return term1 + term2 + term3

def get_errors(X0, X1, classifier_func):
    """
    Экспериментальная оценка вероятностей ошибок.
    X0, X1 – массивы признаков (N, d) для каждого класса.
    """
    err0 = 0
    for x in X0:
        if classifier_func(x) != 0:
            err0 += 1
    p0 = err0 / len(X0)
    
    err1 = 0
    for x in X1:
        if classifier_func(x) != 1:
            err1 += 1
    p1 = err1 / len(X1)
    
    return p0, p1

# ================== Задание 1 ==================
def task1_bayes_equal_B():
    print("Выполнение задания 1: Байесовский классификатор (равные B, равные P)")
    
    # Загрузка данных из файлов, созданных lab1.py
    X0 = np.load("normal_eq_1.npy")   # (N,2)
    X1 = np.load("normal_eq_2.npy")
    
    # Истинные параметры распределений (известны из постановки)
    M0 = np.array([0.0, 1.0])
    M1 = np.array([1.0, -1.0])
    B = np.array([[1.0, 0.4], [0.4, 1.0]])
    
    def classify(x):
        d0, d1 = bayes_normal_equal_B(x, M0, M1, B)
        return 0 if d0 > d1 else 1
    
    p0_exp, p1_exp = get_errors(X0, X1, classify)
    
    # Аналитическая оценка (через расстояние Махаланобиса)
    rho = mahalanobis_distance(M0, M1, B)
    p_err_analyt = 1 - norm.cdf(0.5 * np.sqrt(rho))
    
    print(f"\tЭкспериментальные ошибки: p0={p0_exp:.4f}, p1={p1_exp:.4f}")
    print(f"\tОбщий риск классификации (аналитически): p={p_err_analyt:.4f}")
    
    # Визуализация
    plt.figure(figsize=(8, 6))
    plt.scatter(X0[:, 0], X0[:, 1], s=5, alpha=0.5, label="Класс 0")
    plt.scatter(X1[:, 0], X1[:, 1], s=5, alpha=0.5, label="Класс 1")
    
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            d0, d1 = bayes_normal_equal_B(np.array([xx[i, j], yy[i, j]]), M0, M1, B)
            Z[i, j] = d1 - d0
            
    plt.contour(xx, yy, Z, levels=[0], colors='black')
    plt.title("Байесовская граница (равные B)")
    plt.legend()
    plt.savefig("task1.png")
    plt.close()

# ================== Задание 2 ==================
def task2_minimax_neyman():
    print("Выполнение задания 2: Минимаксный и Неймана-Пирсона")
    
    # Загрузка данных
    X0 = np.load("normal_eq_1.npy")
    X1 = np.load("normal_eq_2.npy")
    
    # Истинные параметры
    M0 = np.array([0.0, 1.0])
    M1 = np.array([1.0, -1.0])
    B = np.array([[1.0, 0.4], [0.4, 1.0]])
    B_inv = inv(B)
    
    # Неймана-Пирсона (p0* = 0.05)
    p0_target = 0.05
    rho = mahalanobis_distance(M0, M1, B)
    # Порог для логарифма отношения правдоподобия
    h = np.sqrt(rho) * norm.ppf(1 - p0_target) - 0.5 * rho
    
    def classify_np(x):
        L = (M1 - M0).T @ B_inv @ x - 0.5 * (M1.T @ B_inv @ M1 - M0.T @ B_inv @ M0)
        return 1 if L > h else 0
    
    p0_exp, p1_exp = get_errors(X0, X1, classify_np)
    print(f"\tНейман-Пирсон (цель p0=0.05): эксп p0={p0_exp:.4f}, p1={p1_exp:.4f}")
    
    # Визуализация (сравнение с байесовской границей)
    plt.figure(figsize=(8, 6))
    plt.scatter(X0[:, 0], X0[:, 1], s=5, alpha=0.5, label="Класс 0")
    plt.scatter(X1[:, 0], X1[:, 1], s=5, alpha=0.5, label="Класс 1")
    
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z_bayes = np.zeros(xx.shape)
    Z_np = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            pos = np.array([xx[i, j], yy[i, j]])
            d0, d1 = bayes_normal_equal_B(pos, M0, M1, B, P0=0.5, P1=0.5)
            Z_bayes[i, j] = d1 - d0
            L = (M1 - M0).T @ B_inv @ pos - 0.5 * (M1.T @ B_inv @ M1 - M0.T @ B_inv @ M0)
            Z_np[i, j] = L - h
    
    plt.contour(xx, yy, Z_bayes, levels=[0], colors='blue', linestyles='dashed')
    plt.contour(xx, yy, Z_np, levels=[0], colors='red')
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, linestyle='--', label='Байес'),
        Line2D([0], [0], color='red', lw=2, label='Нейман-Пирсон')
    ]
    plt.legend(handles=legend_elements)
    plt.title("Границы Байеса и Неймана-Пирсона")
    plt.savefig("task2.png")
    plt.close()

# ================== Задание 3 ==================
def task3_unequal_B():
    print("Выполнение задания 3: Байес для 3 классов (разные B)")
    
    # Загрузка данных
    X0 = np.load("normal_3_1.npy")
    X1 = np.load("normal_3_2.npy")
    X2 = np.load("normal_3_3.npy")
    
    # Истинные параметры (из lab1)
    M0 = np.array([0.0, 1.0])
    M1 = np.array([1.0, -1.0])
    M2 = np.array([-2.0, 1.0])
    B0 = np.array([[1.0, 0.1], [0.1, 1.0]])
    B1 = np.array([[1.0, 0.4], [0.4, 1.0]])
    B2 = np.array([[1.0, -0.4], [-0.4, 1.0]])
    P = [1/3, 1/3, 1/3]
    
    def classify(x):
        d0 = bayes_normal_unequal_B(x, M0, B0, P[0])
        d1 = bayes_normal_unequal_B(x, M1, B1, P[1])
        d2 = bayes_normal_unequal_B(x, M2, B2, P[2])
        return np.argmax([d0, d1, d2])
    
    # Оценка ошибок
    N = len(X0)
    errs = [0, 0, 0]
    for x in X0:
        if classify(x) != 0: errs[0] += 1
    for x in X1:
        if classify(x) != 1: errs[1] += 1
    for x in X2:
        if classify(x) != 2: errs[2] += 1
    
    print(f"\tЭкспериментальные ошибки: p0={errs[0]/N:.4f}, p1={errs[1]/N:.4f}, p2={errs[2]/N:.4f}")
    
    # Относительная погрешность для оценок (при N=1000)
    for i, err in enumerate(errs):
        p_est = err / N
        rel_err = np.sqrt((1 - p_est) / (N * p_est)) if p_est > 0 else np.inf
        print(f"\tОтносительная погрешность для p{i}: {rel_err:.4f}")
    
    # Необходимый объем выборки для погрешности не более 5%
    # ε = sqrt((1-p)/(N p)) → N = (1-p)/(ε² p)
    target_eps = 0.05
    for i, err in enumerate(errs):
        p_est = err / N
        if p_est > 0:
            needed_N = (1 - p_est) / (target_eps**2 * p_est)
            print(f"\tДля p{i} (={p_est:.4f}) объем выборки для ε≤5%: N≈{needed_N:.0f}")
    
    # Визуализация
    plt.figure(figsize=(8, 6))
    plt.scatter(X0[:, 0], X0[:, 1], s=5, alpha=0.3, label="Класс 0")
    plt.scatter(X1[:, 0], X1[:, 1], s=5, alpha=0.3, label="Класс 1")
    plt.scatter(X2[:, 0], X2[:, 1], s=5, alpha=0.3, label="Класс 2")
    
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
    Z = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            Z[i, j] = classify(np.array([xx[i, j], yy[i, j]]))
    
    plt.contourf(xx, yy, Z, alpha=0.2, cmap='viridis')
    plt.title("Байесовские области для 3 классов (разные B)")
    plt.legend()
    plt.savefig("task3.png")
    plt.close()

# ================== Задание 4 ==================
def task4_binary():
    print("Выполнение задания 4: Бинарные векторы")
    
    # Загрузка данных из файлов lab1
    Xbin_1 = np.load("bin_class1_Sch.npy")   # форма (81, N)
    Xbin_2 = np.load("bin_class2_SM.npy")    # форма (81, N)
    # Транспонируем, чтобы получить (N, 81)
    X0 = Xbin_1.T
    X1 = Xbin_2.T
    
    # Параметры модели (из lab1)
    rep0 = representative_Sch()  # (9,9) -> flatten
    rep1 = representative_SM()
    p = 0.3
    
    r0 = rep0.flatten()
    r1 = rep1.flatten()
    
    def get_prob(rep_bit, p_flip):
        return (1 - p_flip) if rep_bit == 1 else p_flip
    
    probs0 = np.array([get_prob(r, p) for r in r0])
    probs1 = np.array([get_prob(r, p) for r in r1])
    
    def classify_bin(x):
        # Логарифмы правдоподобия
        log_f0 = np.sum(x * np.log(probs0) + (1 - x) * np.log(1 - probs0))
        log_f1 = np.sum(x * np.log(probs1) + (1 - x) * np.log(1 - probs1))
        return 0 if log_f0 > log_f1 else 1
    
    p0_exp, p1_exp = get_errors(X0, X1, classify_bin)
    print(f"\tЭкспериментальные ошибки: p0={p0_exp:.4f}, p1={p1_exp:.4f}")
    
    # Аналитическая оценка через ЦПТ
    def get_stats(target_probs, p0_vec, p1_vec):
        L1 = np.log(p1_vec / p0_vec)
        L0 = np.log((1 - p1_vec) / (1 - p0_vec))
        e_i = target_probs * L1 + (1 - target_probs) * L0
        var_i = target_probs * (L1**2) + (1 - target_probs) * (L0**2) - (e_i**2)
        return np.sum(e_i), np.sqrt(np.sum(var_i))
    
    m0, s0 = get_stats(probs0, probs0, probs1)
    m1, s1 = get_stats(probs1, probs0, probs1)
    
    # Порог λ = ln(P(Ω0)/P(Ω1)) = ln(1) = 0 (априорные вероятности равны)
    lamb = 0.0
    p0_analyt = 1 - norm.cdf((lamb - m0) / s0)
    p1_analyt = norm.cdf((lamb - m1) / s1)
    
    print(f"\tАналитические оценки (ЦПТ): p0={p0_analyt:.4f}, p1={p1_analyt:.4f}")

# ================== Главная функция ==================
if __name__ == "__main__":
    task1_bayes_equal_B()
    task2_minimax_neyman()
    task3_unequal_B()
    task4_binary()