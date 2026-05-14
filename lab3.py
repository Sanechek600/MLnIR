from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, det

from utility import simulate_normal, estimate_params

OUT = Path(__file__).resolve().parent / 'out'
OUT.mkdir(parents=True, exist_ok=True)
LOG_DIR = OUT / 'logs'
LOG_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(7)
N = 200

M1 = np.array([1.0, 0.0])
M2 = np.array([1.0, -1.0])

B_equal = np.array([[1.0, 0.4],
                    [0.4, 1.0]])
B1 = np.array([[1.0, 0.1],
               [0.1, 1.0]])
B2 = np.array([[1.0, -0.9],
               [-0.9, 1.0]])


def ensure_data() -> Dict[str, np.ndarray]:
    files = {
        'eq1': LOG_DIR / 'normal_eq_1.npy',
        'eq2': LOG_DIR / 'normal_eq_2.npy',
        '3_1': LOG_DIR / 'normal_3_1.npy',
        '3_2': LOG_DIR / 'normal_3_2.npy',
    }

    if all(p.exists() for p in files.values()):
        return {
            'X1': np.load(files['eq1']),
            'X2': np.load(files['eq2']),
            'X1_3': np.load(files['3_1']),
            'X2_3': np.load(files['3_2']),
        }

    X1 = simulate_normal(M1, B_equal, N)
    X2 = simulate_normal(M2, B_equal, N)
    X1_3 = simulate_normal(M1, B1, N)
    X2_3 = simulate_normal(M2, B2, N)

    np.save(files['eq1'], X1)
    np.save(files['eq2'], X2)
    np.save(files['3_1'], X1_3)
    np.save(files['3_2'], X2_3)

    return {'X1': X1, 'X2': X2, 'X1_3': X1_3, 'X2_3': X2_3}


def classify_linear(X: np.ndarray, w: np.ndarray, w0: float) -> np.ndarray:
    return (X @ w + w0 >= 0).astype(int)


def classification_error(X0: np.ndarray, X1: np.ndarray, w: np.ndarray, w0: float) -> float:
    accuracy = 0.5 * (
        np.mean(classify_linear(X0, w, w0) == 0) +
        np.mean(classify_linear(X1, w, w0) == 1)
    )
    return 1.0 - accuracy


def plot_linear_boundary(ax, w, w0, xlim, label, style='-'):
    xs = np.linspace(xlim[0], xlim[1], 200)
    if abs(w[1]) < 1e-12:
        x = -w0 / w[0]
        ax.axvline(x, linestyle=style, label=label)
        return
    ys = -(w[0] * xs + w0) / w[1]
    ax.plot(xs, ys, style, linewidth=2, label=label)

def plot_bayes_quadratic(ax, M1, B1, M2, B2, xlim, ylim, label):
    xs = np.linspace(xlim[0], xlim[1], 400)
    ys = np.linspace(ylim[0], ylim[1], 400)

    Xg, Yg = np.meshgrid(xs, ys)

    grid = np.stack(
        [Xg.ravel(), Yg.ravel()],
        axis=1
    )

    invB1 = inv(B1)
    invB2 = inv(B2)

    g1 = (
        -0.5 * np.log(det(B1))
        -0.5 * np.einsum(
            '...i,ij,...j->...',
            grid - M1,
            invB1,
            grid - M1
        )
    )

    g2 = (
        -0.5 * np.log(det(B2))
        -0.5 * np.einsum(
            '...i,ij,...j->...',
            grid - M2,
            invB2,
            grid - M2
        )
    )

    Z = (g2 - g1).reshape(Xg.shape)

    ax.contour(
        Xg,
        Yg,
        Z,
        levels=[0],
        linewidths=2,
        linestyles='-',
        colors='black'
    )

    ax.plot([], [], color='black', linewidth=2, label=label)

def fisher_linear_params(M0: np.ndarray, B0: np.ndarray, M1: np.ndarray, B1: np.ndarray):
    Sw = B0 + B1
    w = inv(Sw) @ (M1 - M0)
    m0 = w @ M0
    m1 = w @ M1
    w0 = -0.5 * (m1 + m0)
    return w, w0


def mse_linear_params(X0: np.ndarray, X1: np.ndarray):
    Z = np.vstack([
        np.hstack([X1, np.ones((X1.shape[0], 1))]),
        np.hstack([X0, np.ones((X0.shape[0], 1))]),
    ])
    y = np.hstack([
        np.ones(X1.shape[0]),
        -np.ones(X0.shape[0]),
    ])
    W = np.linalg.pinv(Z) @ y
    return W[:-1], W[-1]


def rm_linear_params(X0: np.ndarray, X1: np.ndarray, alpha0=0.4, beta=0.7, epochs=25, W0=None, seed=7):
    rng_local = np.random.default_rng(seed)
    Z = np.vstack([
        np.hstack([X1, np.ones((X1.shape[0], 1))]),
        np.hstack([X0, np.ones((X0.shape[0], 1))]),
    ])
    y = np.hstack([
        np.ones(X1.shape[0]),
        -np.ones(X0.shape[0]),
    ])
    n = Z.shape[1]
    W = np.zeros(n) if W0 is None else W0.astype(float).copy()
    hist = []
    t = 1
    idx = np.arange(len(Z))
    for _ in range(epochs):
        rng_local.shuffle(idx)
        for i in idx:
            alpha = alpha0 / (t ** beta)
            z = Z[i]
            yi = y[i]
            W = W + alpha * z * (yi - float(W @ z))
            hist.append(W.copy())
            t += 1
    return W[:-1], W[-1], np.asarray(hist)


def rm_history_errors(history: np.ndarray, X0: np.ndarray, X1: np.ndarray):
    Z = np.vstack([
        np.hstack([X1, np.ones((X1.shape[0], 1))]),
        np.hstack([X0, np.ones((X0.shape[0], 1))]),
    ])
    y = np.hstack([
        np.ones(X1.shape[0]),
        -np.ones(X0.shape[0]),
    ])
    errs = []
    for W in history:
        pred = np.where(Z @ W >= 0, 1.0, -1.0)
        errs.append(float(np.mean(pred != y)))
    return np.asarray(errs)


def lab3(data: Dict[str, np.ndarray]):
    X1, X2 = data['X1'], data['X2']
    X1_3, X2_3 = data['X1_3'], data['X2_3']

    w_b = inv(B_equal) @ (M2 - M1)
    w0_b = -0.5 * (M2 @ inv(B_equal) @ M2 - M1 @ inv(B_equal) @ M1)

    M1_hat, B1_hat = estimate_params(X1)
    M2_hat, B2_hat = estimate_params(X2)
    w_f_eq, w0_f_eq = fisher_linear_params(M1, B_equal, M2, B_equal)
    w_mse_eq, w0_mse_eq = mse_linear_params(X1, X2)
    w_rm_eq, w0_rm_eq, hist_eq = rm_linear_params(X1, X2, alpha0=0.35, beta=0.7, epochs=30, seed=11)
    err_hist_eq = rm_history_errors(hist_eq, X1, X2)

    M1_hat3, B1_hat3 = estimate_params(X1_3)
    M2_hat3, B2_hat3 = estimate_params(X2_3)
    w_f_une, w0_f_une = fisher_linear_params(M1, B1, M2, B2)
    w_mse_une, w0_mse_une = mse_linear_params(X1_3, X2_3)
    w_rm_une, w0_rm_une, hist_une = rm_linear_params(X1_3, X2_3, alpha0=0.35, beta=0.7, epochs=30, seed=13)
    err_hist_une = rm_history_errors(hist_une, X1_3, X2_3)

    bayes_pred_eq_0 = classify_linear(X1, w_b, w0_b)
    bayes_pred_eq_1 = classify_linear(X2, w_b, w0_b)
    bayes_err_eq = 1.0 - 0.5 * (np.mean(bayes_pred_eq_0 == 0) + np.mean(bayes_pred_eq_1 == 1))
    def bayes_predict_une(X):
        g1 = -0.5 * np.log(det(B1)) - 0.5 * np.einsum('...i,ij,...j->...', X - M1, inv(B1), X - M1)
        g2 = -0.5 * np.log(det(B2)) - 0.5 * np.einsum('...i,ij,...j->...', X - M2, inv(B2), X - M2)
        return (g2 >= g1).astype(int)

    bayes_pred_une_0 = bayes_predict_une(X1_3)
    bayes_pred_une_1 = bayes_predict_une(X2_3)
    bayes_err_une = 1.0 - 0.5 * (np.mean(bayes_pred_une_0 == 0) + np.mean(bayes_pred_une_1 == 1))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X1[:, 0], X1[:, 1], s=12, alpha=0.6, label='Класс Ω0')
    ax.scatter(X2[:, 0], X2[:, 1], s=12, alpha=0.6, label='Класс Ω1')
    plot_linear_boundary(ax, w_b, w0_b, (-5.5, 4.5), 'Байес', '-')
    plot_linear_boundary(ax, w_f_eq, w0_f_eq, (-5.5, 4.5), 'Фишер', '--')
    plot_linear_boundary(ax, w_mse_eq, w0_mse_eq, (-5.5, 4.5), 'МСКО', ':')
    plot_linear_boundary(ax, w_rm_eq, w0_rm_eq, (-5.5, 4.5), 'Роббинс–Монро', '-.')
    ax.set_title('Лабораторная 3: равные корреляционные матрицы')
    ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.grid(True); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / 'lab3_equal_linear.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X1_3[:, 0], X1_3[:, 1], s=12, alpha=0.6, label='Класс Ω0')
    ax.scatter(X2_3[:, 0], X2_3[:, 1], s=12, alpha=0.6, label='Класс Ω1')
    plot_linear_boundary(ax, w_f_une, w0_f_une, (-5.5, 4.5), 'Фишер', '--')
    plot_linear_boundary(ax, w_mse_une, w0_mse_une, (-5.5, 4.5), 'МСКО', ':')
    plot_linear_boundary(ax, w_rm_une, w0_rm_une, (-5.5, 4.5), 'Роббинс–Монро', '-.')
    plot_bayes_quadratic(ax, M1, B1, M2, B2, (-5.5, 4.5), (-5.5, 4.5), 'Байес')
    ax.set_title('Лабораторная 3: неравные корреляционные матрицы')
    ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.grid(True); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / 'lab3_unequal_linear.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(err_hist_eq, label='Равные ковариации')
    ax.plot(err_hist_une, label='Неравные ковариации')
    ax.set_xlabel('Итерация'); ax.set_ylabel('Доля ошибок на обучении')
    ax.set_title('Сходимость процедуры Роббинса–Монро')
    ax.grid(True); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / 'lab3_rm_convergence.png', dpi=160); plt.close(fig)

    results = {
        'equal_cov': {
            'bayes_train_err': bayes_err_eq,
            'fisher_train_err': classification_error(X1, X2, w_f_eq, w0_f_eq),
            'mse_train_err': classification_error(X1, X2, w_mse_eq, w0_mse_eq),
            'rm_train_err': classification_error(X1, X2, w_rm_eq, w0_rm_eq),
            'fisher_params': {'w': w_f_eq.tolist(), 'w0': float(w0_f_eq)},
            'mse_params': {'w': w_mse_eq.tolist(), 'w0': float(w0_mse_eq)},
            'rm_params': {'w': w_rm_eq.tolist(), 'w0': float(w0_rm_eq)},
        },
        'unequal_cov': {
            'bayes_train_err': bayes_err_une,
            'fisher_train_err': classification_error(X1_3, X2_3, w_f_une, w0_f_une),
            'mse_train_err': classification_error(X1_3, X2_3, w_mse_une, w0_mse_une),
            'rm_train_err': classification_error(X1_3, X2_3, w_rm_une, w0_rm_une),
            'fisher_params': {'w': w_f_une.tolist(), 'w0': float(w0_f_une)},
            'mse_params': {'w': w_mse_une.tolist(), 'w0': float(w0_mse_une)},
            'rm_params': {'w': w_rm_une.tolist(), 'w0': float(w0_rm_une)},
        },
    }
    return results


def write_report_text(lab3_res):
    lines = []
    lines.append('Лабораторная 3')
    lines.append('---------------')
    e = lab3_res['equal_cov']
    lines.append(f"Равные ковариации: P(ош) Bayes={e['bayes_train_err']:.4f}, Fisher={e['fisher_train_err']:.4f}, MSE={e['mse_train_err']:.4f}, RM={e['rm_train_err']:.4f}")
    u = lab3_res['unequal_cov']
    lines.append(f"Неравные ковариации: P(ош) Bayes={u['bayes_train_err']:.4f}, Fisher={u['fisher_train_err']:.4f}, MSE={u['mse_train_err']:.4f}, RM={u['rm_train_err']:.4f}")
    lines.append('')
    lines.append('Параметры классификаторов:')
    lines.append(f"Равные ковариации, Фишер: w={e['fisher_params']['w']}, w0={e['fisher_params']['w0']:.6f}")
    lines.append(f"Равные ковариации, МСКО: w={e['mse_params']['w']}, w0={e['mse_params']['w0']:.6f}")
    lines.append(f"Равные ковариации, Роббинс–Монро: w={e['rm_params']['w']}, w0={e['rm_params']['w0']:.6f}")
    lines.append(f"Неравные ковариации, Фишер: w={u['fisher_params']['w']}, w0={u['fisher_params']['w0']:.6f}")
    lines.append(f"Неравные ковариации, МСКО: w={u['mse_params']['w']}, w0={u['mse_params']['w0']:.6f}")
    lines.append(f"Неравные ковариации, Роббинс–Монро: w={u['rm_params']['w']}, w0={u['rm_params']['w0']:.6f}")
    lines.append('')
    lines.append('Файлы графиков:')
    lines.append('lab3_equal_linear.png')
    lines.append('lab3_unequal_linear.png')
    lines.append('lab3_rm_convergence.png')
    (OUT / 'results.txt').write_text('\n'.join(lines), encoding='utf-8')


def main():
    data = ensure_data()
    lab3_res = lab3(data)
    write_report_text(lab3_res)
    print(f'Готово. Результаты сохранены в: {OUT}')


if __name__ == '__main__':
    main()

# =============================================================================
# Аналитическая часть
#
# Лабораторная 3.
# 1) Критерий Фишера:
#    w = S_w^{-1}(m_1 - m_0),  где S_w = B_0 + B_1.
#    Граница: w^T x + w_0 = 0.
#
# 2) Минимизация среднеквадратичной ошибки:
#    J(W) = (1/N) Σ (y_i - W^T z_i)^2,
#    W = (Z^T Z)^{-1} Z^T y
#    (или псевдообратная матрица при вырожденности), где z_i = [x_i, 1]^T.
#
# 3) Процедура Роббинса–Монро:
#    W_{k+1} = W_k + α_k z_k (y_k - W_k^T z_k),
#    α_k = α_0 / k^β, 0 < β ≤ 1.
# =============================================================================
