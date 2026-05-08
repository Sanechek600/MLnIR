import numpy as np
import matplotlib.pyplot as plt

from utility import *

LOG_PATH = ""
np.random.seed(0)  # чтобы результаты воспроизводились
N = 1000

M1 = np.array([0.0, 1.0])
M2 = np.array([1.0, -1.0])
M3 = np.array([-2.0, 1.0])
B1 = np.array([[1.0, 0.1],
               [0.1, 1.0]])

B2 = np.array([[1.0, 0.4],
               [0.4, 1.0]])

B3 = np.array([[1.0, -0.4],
               [-0.4, 1.0]])
B_equal = np.array([[1.0, 0.4],
                    [0.4, 1.0]])

X1 = simulate_normal(M1, B_equal, N)
X2 = simulate_normal(M2, B_equal, N)

# Сохранение выборок
np.save(LOG_PATH + "normal_eq_1.npy", X1)
np.save(LOG_PATH + "normal_eq_2.npy", X2)

# График
plt.figure(figsize=(6, 6))
plt.scatter(X1[:, 0], X1[:, 1], s=10, alpha=0.6, label="Класс 1")
plt.scatter(X2[:, 0], X2[:, 1], s=10, alpha=0.6, label="Класс 2")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title(f"Две выборки N={N} с равными корреляционными матрицами")
plt.grid(True)
plt.tight_layout()
plt.show()

M1_est, B1_est = estimate_params(X1)
M2_est, B2_est = estimate_params(X2)

rho_b_eq = bhattacharyya_distance(M1_est, B1_est, M2_est, B2_est)
rho_m_eq = mahalanobis_distance(M1_est, M2_est, (B1_est + B2_est) / 2)

print("== Равные корреляционные матрицы ==")
print("Исходные M1:", M1, "M2:", M2)
print("Оценки M1^:", M1_est, "M2^:", M2_est)
print("Исходные B_equal:", B_equal)
print("Оценки B1^:\n", B1_est)
print("Оценки B2^:\n", B2_est)
print("Расстояние Бхатачария:", rho_b_eq)
print("Расстояние Махаланобиса:", rho_m_eq)

#---Для 3 выборок с разными корреляционными матрицами---
X1_3 = simulate_normal(M1, B1, N)
X2_3 = simulate_normal(M2, B2, N)
X3_3 = simulate_normal(M3, B3, N)

np.save(LOG_PATH + "normal_3_1.npy", X1_3)
np.save(LOG_PATH + "normal_3_2.npy", X2_3)
np.save(LOG_PATH + "normal_3_3.npy", X3_3)

plt.figure(figsize=(6, 6))
plt.scatter(X1_3[:, 0], X1_3[:, 1], s=10, alpha=0.6, label="Класс 1")
plt.scatter(X2_3[:, 0], X2_3[:, 1], s=10, alpha=0.6, label="Класс 2")
plt.scatter(X3_3[:, 0], X3_3[:, 1], s=10, alpha=0.6, label="Класс 3")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title(f"Три выборки N={N} с разными корреляционными матрицами")
plt.grid(True)
plt.tight_layout()
plt.show()

M1_3_est, B1_est_3 = estimate_params(X1_3)
M2_3_est, B2_est_3 = estimate_params(X2_3)
M3_3_est, B3_est_3 = estimate_params(X3_3)

pairs = [
    ("1-2", M1_3_est, B1_est_3, M2_3_est, B2_est_3),
    ("1-3", M1_3_est, B1_est_3, M3_3_est, B3_est_3),
    ("2-3", M2_3_est, B2_est_3, M3_3_est, B3_est_3),
]

print("== Разные корреляционные матрицы ==")

print("Исходные M1:", M1, "M2:", M2, "M3:", M3)
print("Оценки M1^:", M1_3_est, "M2^:", M2_3_est, "M3^:", M3_3_est)
print("Исходная B1^:\n", B1)
print("Исходная B2^:\n", B2)
print("Исходная B3^:\n", B3)
print("Оценки B1^:\n", B1_est_3)
print("Оценки B2^:\n", B2_est_3)
print("Оценки B3^:\n", B3_est_3)

for name, Ma, Ba, Mb, Bb in pairs:
    rho_b = bhattacharyya_distance(Ma, Ba, Mb, Bb)
    print(f"Пара {name}: расстояние Бхатачария = {rho_b}")

print("== Бинарные векторы ==")

p = 0.3

SCH = [
        [1, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]

SM = [
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0],
    ]

rep_1 = representative(SCH)
rep_2 = representative(SM)

plot_binary_vector(rep_1, "Представитель класса 1: 'Щ' (9x9)")
plot_binary_vector(rep_2, "Представитель класса 2: 'Ь' (9x9)")

Xbin_1 = generate_binary_samples(rep_1, N=N, p=p)
Xbin_2 = generate_binary_samples(rep_2, N=N, p=p)

np.save("bin_class1_Sch.npy", Xbin_1)
np.save("bin_class2_SM.npy", Xbin_2)

rep1_flat = rep_1.reshape(-1, 1)
rep2_flat = rep_2.reshape(-1, 1)
changed_1 = np.mean(Xbin_1 != rep1_flat)
changed_2 = np.mean(Xbin_2 != rep2_flat)

print(f"p задано: {p}")
print(f"Фактическая доля измененных битов для 'Щ': {changed_1:.4f}")
print(f"Фактическая доля измененных битов для 'Ь': {changed_2:.4f}")
