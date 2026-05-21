
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Callable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, det
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC

try:
    from qpsolvers import solve_qp, available_solvers
    QPSOLVERS_AVAILABLE = True
except Exception:
    solve_qp = None
    available_solvers = []
    QPSOLVERS_AVAILABLE = False
    print("qpsolvers не найден. QP-часть будет пропущена.")
    print("Установите пакет: pip install qpsolvers")
    print("И один backend-решатель, например: pip install clarabel")

# ---------------------------------------------------------------------
# Папки для вывода
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "out"
OUT.mkdir(parents=True, exist_ok=True)
LOG_DIR = OUT / "logs"
LOG_DIR.mkdir(exist_ok=True)

np.random.seed(42)

# ---------------------------------------------------------------------
# Исходные данные
# ---------------------------------------------------------------------
N = 200
N_ERR = 50000

# Линейно разделимый случай
N_SEP = 100
M_SEP_1 = np.array([-2.0, 1.5])
M_SEP_2 = np.array([2.0, -1.5])
B_SEP = np.array([
    [0.15, 0.0],
    [0.0, 0.15],
])

# Данные "как из ЛР1" — используются для линейно неразделимого случая
M1 = np.array([-1.0, 1.0])
M2 = np.array([0.0, 1.0])
M3 = np.array([-1.0, -1.0])

B1 = np.array([
    [1.0, -0.85],
    [-0.85, 1.0],
])
B2 = np.array([
    [0.6, 0.0],
    [0.0, 0.6],
])
B3 = np.array([
    [1.0, 0.85],
    [0.85, 1.0],
])

# ---------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------
def choose_qp_solver() -> Optional[str]:
    preferred = ("clarabel", "osqp", "proxqp", "quadprog", "scs")
    for solver in preferred:
        if solver in available_solvers:
            return solver
    return available_solvers[0] if available_solvers else None


QP_SOLVER = choose_qp_solver()
if QPSOLVERS_AVAILABLE:
    print("Доступные QP-решатели:", available_solvers)
    print("Используемый QP-решатель:", QP_SOLVER)


def save_npy(name: str, arr: np.ndarray) -> None:
    np.save(LOG_DIR / name, arr)


def classification_error(X0: np.ndarray, X1: np.ndarray, model) -> float:
    accuracy = 0.5 * (
        np.mean(model.predict(X0) == -1) +
        np.mean(model.predict(X1) == 1)
    )
    return 1.0 - float(accuracy)


def error_probability(model, X: np.ndarray, y: np.ndarray) -> float:
    y_pred = model.predict(X)
    return 1.0 - float(accuracy_score(y, y_pred))


def shown_support_vectors(model, max_count: int = 80, seed: int = 42):
    if not hasattr(model, "support_vectors_"):
        return None
    support_vectors = getattr(model, "support_vectors_", None)
    if support_vectors is None:
        return None
    if len(support_vectors) <= max_count:
        return support_vectors
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(support_vectors), size=max_count, replace=False)
    return support_vectors[idx]


def plot_decision_function(
    model,
    X_a: np.ndarray,
    X_b: np.ndarray,
    title: str,
    save_path: Path,
    max_support_plot: int = 80,
):
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(X_a[:, 0], X_a[:, 1], label="Класс +1", alpha=0.75)
    ax.scatter(X_b[:, 0], X_b[:, 1], label="Класс -1", alpha=0.75)

    x_min = min(X_a[:, 0].min(), X_b[:, 0].min()) - 1
    x_max = max(X_a[:, 0].max(), X_b[:, 0].max()) + 1
    y_min = min(X_a[:, 1].min(), X_b[:, 1].min()) - 1
    y_max = max(X_a[:, 1].max(), X_b[:, 1].max()) + 1

    xx = np.linspace(x_min, x_max, 250)
    yy = np.linspace(y_min, y_max, 250)
    YY, XX = np.meshgrid(yy, xx)
    grid = np.vstack([XX.ravel(), YY.ravel()]).T

    Z = model.decision_function(grid).reshape(XX.shape)

    ax.contour(
        XX,
        YY,
        Z,
        levels=[-1, 0, 1],
        linestyles=["--", "-", "--"],
        colors="black",
    )

    sv = shown_support_vectors(model, max_count=max_support_plot)
    if sv is not None and len(sv) > 0:
        ax.scatter(
            sv[:, 0],
            sv[:, 1],
            s=100,
            facecolors="none",
            edgecolors="black",
            label=f"Опорные векторы, показано {len(sv)}",
        )

    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def linear_kernel(X: np.ndarray, Z: np.ndarray):
    return X @ Z.T


def poly2_kernel(X: np.ndarray, Z: np.ndarray):
    return (X @ Z.T + 1) ** 2


def rbf_kernel(X: np.ndarray, Z: np.ndarray, gamma: float = 1.0):
    X_norm = np.sum(X ** 2, axis=1)[:, None]
    Z_norm = np.sum(Z ** 2, axis=1)[None, :]
    dist2 = X_norm + Z_norm - 2 * (X @ Z.T)
    return np.exp(-gamma * dist2)


def sigmoid_kernel(X: np.ndarray, Z: np.ndarray, gamma: float = 0.1, coef0: float = -1):
    return np.tanh(gamma * (X @ Z.T) + coef0)


def gaussian_score(X: np.ndarray, M: np.ndarray, B: np.ndarray, prior: float = 0.5) -> np.ndarray:
    Bi = inv(B)
    return (
        math.log(prior)
        - 0.5 * np.log(det(B))
        - 0.5 * np.einsum("...i,ij,...j->...", X - M, Bi, X - M)
    )


class GaussianBayesClassifier:
    def __init__(self, M0: np.ndarray, B0: np.ndarray, M1_: np.ndarray, B1_: np.ndarray, prior0: float = 0.5, prior1: float = 0.5):
        self.M0 = np.asarray(M0)
        self.B0 = np.asarray(B0)
        self.M1 = np.asarray(M1_)
        self.B1 = np.asarray(B1_)
        self.prior0 = prior0
        self.prior1 = prior1
        self.support_vectors_ = None

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        g0 = gaussian_score(X, self.M0, self.B0, self.prior0)
        g1 = gaussian_score(X, self.M1, self.B1, self.prior1)
        return g1 - g0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0, 1, -1)


def plot_bayes_boundary(
    ax,
    M0: np.ndarray,
    B0: np.ndarray,
    M1_: np.ndarray,
    B1_: np.ndarray,
    xlim,
    ylim,
    label: str,
    style: str = "-",
):
    xs = np.linspace(xlim[0], xlim[1], 400)
    ys = np.linspace(ylim[0], ylim[1], 400)
    Xg, Yg = np.meshgrid(xs, ys)
    grid = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
    Z = (
        gaussian_score(grid, M1_, B1_)
        - gaussian_score(grid, M0, B0)
    ).reshape(Xg.shape)
    ax.contour(Xg, Yg, Z, levels=[0], linewidths=2, linestyles=style, colors="black")
    ax.plot([], [], color="black", linewidth=2, linestyle=style, label=label)


def print_model_info(model, X: np.ndarray, y: np.ndarray, name: str):
    p = error_probability(model, X, y)

    print(name)
    print("Экспериментальная вероятность ошибки p =", p)

    if hasattr(model, "coef_"):
        print("w =", model.coef_[0])
        print("w_N =", model.intercept_[0])

    if hasattr(model, "support_vectors_") and getattr(model, "support_vectors_", None) is not None:
        print("Количество опорных векторов =", len(model.support_vectors_))

    print()


def to_Nx2(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    return X.T if X.shape[0] == 2 else X


def load_or_generate(filename: str, generator: Callable[[], np.ndarray]) -> np.ndarray:
    path = LOG_DIR / filename
    X = generator()
    np.save(path, X)
    return X


# ---------------------------------------------------------------------
# QP-решение для линейного SVM
# ---------------------------------------------------------------------
def solve_linear_svm_qp(X: np.ndarray, y: np.ndarray, C: Optional[float] = None):
    if not QPSOLVERS_AVAILABLE:
        raise RuntimeError("qpsolvers не установлен.")

    n_samples = X.shape[0]
    K = X @ X.T
    P = np.outer(y, y) * K
    P = (P + P.T) / 2
    P = P + 1e-8 * np.eye(n_samples)

    q = -np.ones(n_samples)
    A = y.reshape(1, -1).astype(float)
    b = np.array([0.0])

    if C is None:
        G = -np.eye(n_samples)
        h = np.zeros(n_samples)
    else:
        G = np.vstack((-np.eye(n_samples), np.eye(n_samples)))
        h = np.hstack((np.zeros(n_samples), C * np.ones(n_samples)))

    alpha = solve_qp(P, q, G, h, A, b, solver=QP_SOLVER)
    if alpha is None:
        raise ValueError("solve_qp не нашел решение для линейного SVM.")

    alpha = np.asarray(alpha)
    support_mask = alpha > 1e-5
    w = np.sum((alpha * y)[:, None] * X, axis=0)

    if C is None:
        margin_mask = support_mask
    else:
        margin_mask = (alpha > 1e-5) & (alpha < C - 1e-5)

    if np.any(margin_mask):
        b_value = np.mean(y[margin_mask] - X[margin_mask] @ w)
    else:
        b_value = np.mean(y[support_mask] - X[support_mask] @ w)

    return alpha, w, float(b_value), support_mask


class QPLinearSVM:
    def __init__(self, X_train, y_train, w, b, support_mask):
        self.X_train = X_train
        self.y_train = y_train
        self.coef_ = np.array([w])
        self.intercept_ = np.array([b])
        self.support_vectors_ = X_train[support_mask]

    def decision_function(self, X):
        return X @ self.coef_[0] + self.intercept_[0]

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


# ---------------------------------------------------------------------
# QP-решение для ядерного SVM
# ---------------------------------------------------------------------
def solve_kernel_svm_qp(X: np.ndarray, y: np.ndarray, kernel_func: Callable, C: float = 1.0):
    if not QPSOLVERS_AVAILABLE:
        raise RuntimeError("qpsolvers не установлен.")

    n = X.shape[0]
    K = kernel_func(X, X)

    P = np.outer(y, y) * K
    P = (P + P.T) / 2
    P = P + 1e-8 * np.eye(n)

    q = -np.ones(n)
    G = np.vstack((-np.eye(n), np.eye(n))).astype(float)
    h = np.hstack((np.zeros(n), C * np.ones(n))).astype(float)
    A = y.reshape(1, -1).astype(float)
    b = np.array([0.0])

    alpha = solve_qp(P.astype(float), q.astype(float), G, h, A, b, solver=QP_SOLVER)
    if alpha is None:
        raise ValueError("solve_qp не нашел решение для ядерного SVM.")

    alpha = np.asarray(alpha)
    support_mask = alpha > 1e-5
    margin_mask = (alpha > 1e-5) & (alpha < C - 1e-5)

    if np.any(margin_mask):
        idx = np.where(margin_mask)[0]
    else:
        idx = np.where(support_mask)[0]

    b_values = []
    for i in idx:
        K_i = kernel_func(X, X[i:i+1]).reshape(-1)
        s = np.sum(alpha * y * K_i)
        b_values.append(y[i] - s)

    b_value = float(np.mean(b_values))
    return alpha, b_value, support_mask


class QPKernelSVM:
    def __init__(self, X_train, y_train, alpha, b, kernel_func):
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha
        self.b = b
        self.kernel_func = kernel_func
        self.support_vectors_ = X_train[alpha > 1e-5]

    def decision_function(self, X):
        K = self.kernel_func(self.X_train, X)
        return np.sum((self.alpha * self.y_train)[:, None] * K, axis=0) + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


# ---------------------------------------------------------------------
# 1. Генерация данных
# ---------------------------------------------------------------------
def generate_linear_separable_data():
    X_sep_1 = np.random.multivariate_normal(M_SEP_1, B_SEP, N_SEP)
    X_sep_2 = np.random.multivariate_normal(M_SEP_2, B_SEP, N_SEP)
    X_sep = np.vstack((X_sep_1, X_sep_2))
    y_sep = np.hstack((np.ones(N_SEP), -np.ones(N_SEP)))

    save_npy("svm_sep_class1.npy", X_sep_1)
    save_npy("svm_sep_class2.npy", X_sep_2)
    save_npy("svm_sep_X.npy", X_sep)
    save_npy("svm_sep_y.npy", y_sep)

    return X_sep_1, X_sep_2, X_sep, y_sep


def generate_neq_data():
    X1 = np.random.multivariate_normal(M1, B1, N)
    X2 = np.random.multivariate_normal(M2, B2, N)
    X3 = np.random.multivariate_normal(M3, B3, N)

    save_npy("gauss_neq_class1_M1_B1.npy", X1)
    save_npy("gauss_neq_class2_M2_B2.npy", X2)
    save_npy("gauss_neq_class3_M3_B3.npy", X3)

    return to_Nx2(X1), to_Nx2(X2), to_Nx2(X3)


# ---------------------------------------------------------------------
# 2. Запуск экспериментов
# ---------------------------------------------------------------------
def run_lab4():
    results: Dict[str, object] = {}

    # ------------------------------------------
    # Линейно разделимые классы
    # ------------------------------------------
    print("1) Синтез линейно разделимых выборок")
    X_sep_1, X_sep_2, X_sep, y_sep = generate_linear_separable_data()
    print("Размер X_sep:", X_sep.shape)
    print("Размер y_sep:", y_sep.shape)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(X_sep_1[:, 0], X_sep_1[:, 1], label="Класс +1")
    ax.scatter(X_sep_2[:, 0], X_sep_2[:, 1], label="Класс -1")
    ax.set_title("Синтезированные линейно разделимые выборки")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "lab4_sep_data.png", dpi=160)
    plt.close(fig)

    bayes_sep = GaussianBayesClassifier(M_SEP_1, B_SEP, M_SEP_2, B_SEP)

    print("\n2) SVM для линейно разделимых классов")

    if QPSOLVERS_AVAILABLE:
        alpha_sep, w_sep_qp, b_sep_qp, support_sep = solve_linear_svm_qp(X_sep, y_sep, C=None)
        qp_svm_sep = QPLinearSVM(X_sep, y_sep, w_sep_qp, b_sep_qp, support_sep)

        print_model_info(qp_svm_sep, X_sep, y_sep, "Решение через solve_qp для линейно разделимых классов")
        print("Bayes, линейно разделимые классы")
        print("Экспериментальная вероятность ошибки p =", 1.0 - error_probability(bayes_sep, X_sep, y_sep))
        print()

        plot_decision_function(
            qp_svm_sep,
            X_sep_1,
            X_sep_2,
            "SVM через solve_qp: линейно разделимые классы",
            OUT / "lab4_sep_qp.png",
        )
    else:
        print("Пропущено: qpsolvers не установлен.")

    svc_sep = SVC(kernel="linear", C=1_000_000)
    svc_sep.fit(X_sep, y_sep)

    print_model_info(svc_sep, X_sep, y_sep, "SVC, линейное ядро, линейно разделимые классы")
    print("Bayes, линейно разделимые классы")
    print("Экспериментальная вероятность ошибки p =", 1.0 - error_probability(bayes_sep, X_sep, y_sep))
    print()

    plot_decision_function(
        svc_sep,
        X_sep_1,
        X_sep_2,
        "SVC: линейно разделимые классы",
        OUT / "lab4_sep_svc.png",
    )

    linear_svc_sep = LinearSVC(C=1_000_000, max_iter=20000)
    linear_svc_sep.fit(X_sep, y_sep)

    print_model_info(linear_svc_sep, X_sep, y_sep, "LinearSVC, линейно разделимые классы")
    print()

    plot_decision_function(
        linear_svc_sep,
        X_sep_1,
        X_sep_2,
        "LinearSVC: линейно разделимые классы",
        OUT / "lab4_sep_linearsvc.png",
    )

    # Bayes boundary for the separable case
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(X_sep_1[:, 0], X_sep_1[:, 1], label="Класс +1")
    ax.scatter(X_sep_2[:, 0], X_sep_2[:, 1], label="Класс -1")
    plot_bayes_boundary(ax, M_SEP_1, B_SEP, M_SEP_2, B_SEP, (-5.5, 5.5), (-5.0, 5.0), "Байес")
    ax.set_title("Байесовская граница: линейно разделимые классы")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "lab4_sep_bayes.png", dpi=160)
    plt.close(fig)

    results["separable"] = {
        "bayes_error": 1.0 - float(error_probability(bayes_sep, X_sep, y_sep)),
        "qp": {
            "w": w_sep_qp.tolist() if QPSOLVERS_AVAILABLE else None,
            "b": float(b_sep_qp) if QPSOLVERS_AVAILABLE else None,
            "support_vectors": int(np.sum(support_sep)) if QPSOLVERS_AVAILABLE else None,
            "error": float(error_probability(qp_svm_sep, X_sep, y_sep)) if QPSOLVERS_AVAILABLE else None,
        } if QPSOLVERS_AVAILABLE else None,
        "svc_error": float(error_probability(svc_sep, X_sep, y_sep)),
        "linearsvc_error": float(error_probability(linear_svc_sep, X_sep, y_sep)),
    }

    # ------------------------------------------
    # Линейно неразделимые классы
    # ------------------------------------------
    print("\n3) Выборки для неравных ковариационных матриц")
    X1, X2, X3 = generate_neq_data()

    print("Исходные размеры:")
    print("X1:", X1.shape)
    print("X2:", X2.shape)
    print("X3:", X3.shape)

    X_13 = np.vstack((X1, X3))
    y_13 = np.hstack((np.ones(len(X1)), -np.ones(len(X3))))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(X1[:, 0], X1[:, 1], label="Класс 1")
    ax.scatter(X2[:, 0], X2[:, 1], label="Класс 2")
    ax.scatter(X3[:, 0], X3[:, 1], label="Класс 3")
    ax.set_title("Наши выборки для неравных ковариационных матриц")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "lab4_neq_data.png", dpi=160)
    plt.close(fig)

    bayes_13 = GaussianBayesClassifier(M1, B1, M3, B3)

    print("\n4) SVM для линейно неразделимых классов (классы 1 и 3)")
    print("Bayes, классы 1 и 3")
    print("Экспериментальная вероятность ошибки p =", 1.0 - error_probability(bayes_13, X_13, y_13))
    print()

    QP_TRAIN_PER_CLASS = 100
    X1_qp = X1[:min(QP_TRAIN_PER_CLASS, len(X1))]
    X3_qp = X3[:min(QP_TRAIN_PER_CLASS, len(X3))]
    X_13_qp = np.vstack((X1_qp, X3_qp))
    y_13_qp = np.hstack((np.ones(len(X1_qp)), -np.ones(len(X3_qp))))

    if QPSOLVERS_AVAILABLE:
        for C in [0.1, 1, 10]:
            alpha, w, b_value, support_mask = solve_linear_svm_qp(X_13_qp, y_13_qp, C=C)
            model = QPLinearSVM(X_13_qp, y_13_qp, w, b_value, support_mask)

            err_full = error_probability(model, X_13, y_13)
            print(f"solve_qp, линейный SVM, C={C}")
            print("w =", w)
            print("w_N =", b_value)
            print("Количество опорных векторов на QP-подвыборке =", np.sum(support_mask))
            print("Экспериментальная вероятность ошибки p на полной выборке =", err_full)
            print()

            plot_decision_function(
                model,
                X1,
                X3,
                f"solve_qp: линейный SVM, C={C}",
                OUT / f"lab4_neq_qp_C_{str(C).replace('.', '_')}.png",
            )

            results.setdefault("neq_qp", {})[C] = {
                "w": w.tolist(),
                "b": float(b_value),
                "support_vectors": int(np.sum(support_mask)),
                "error_full": float(err_full),
            }
    else:
        print("Пропущено: qpsolvers не установлен.")

    # Bayes boundary for the non-separable case
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(X1[:, 0], X1[:, 1], label="Класс 1")
    ax.scatter(X3[:, 0], X3[:, 1], label="Класс 3")
    plot_bayes_boundary(ax, M1, B1, M3, B3, (-5.5, 5.5), (-5.5, 5.5), "Байес", style="-")
    ax.set_title("Байесовская граница: классы 1 и 3")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "lab4_neq_bayes.png", dpi=160)
    plt.close(fig)

    # SVC linear, search best C
    C_grid = [0.01, 0.03, 0.3, 1, 3, 10]
    best_C = None
    best_error = 10**9
    best_model = None

    print("\n5) SVC для линейно неразделимых классов")
    for C in C_grid:
        model = SVC(kernel="linear", C=C)
        model.fit(X_13, y_13)
        p = error_probability(model, X_13, y_13)
        print("C =", C, "ошибка p =", p)

        if p < best_error:
            best_error = p
            best_C = C
            best_model = model

    print()
    print("Лучшее значение C =", best_C)
    print("Минимальная ошибка p =", best_error)
    print("Bayes, классы 1 и 3")
    print("Экспериментальная вероятность ошибки p =", 1.0 - error_probability(bayes_13, X_13, y_13))
    print()

    plot_decision_function(
        best_model,
        X1,
        X3,
        f"Лучший линейный SVM: C={best_C}",
        OUT / "lab4_neq_best_svc.png",
    )

    svc_linear_results = {}
    for C in [0.1, 1, 10, best_C]:
        model = SVC(kernel="linear", C=C)
        model.fit(X_13, y_13)

        print_model_info(
            model,
            X_13,
            y_13,
            f"SVC, линейное ядро, C={C}, классы 1 и 3",
        )

        plot_decision_function(
            model,
            X1,
            X3,
            f"SVC: линейный SVM, C={C}",
            OUT / f"lab4_neq_svc_linear_C_{str(C).replace('.', '_')}.png",
        )

        svc_linear_results[C] = {
            "error": float(error_probability(model, X_13, y_13)),
            "support_vectors": int(len(model.support_vectors_)),
        }

    results["neq_linear"] = {
        "bayes_error": 1.0 - float(error_probability(bayes_13, X_13, y_13)),
        "best_C": float(best_C),
        "best_error": float(best_error),
        "svc": svc_linear_results,
    }

    # ------------------------------------------
    # Ядерный SVM через solve_qp
    # ------------------------------------------
    print("\n6) Ядерный SVM через solve_qp")
    kernel_qp_list = [
        ("linear", linear_kernel),
        ("poly2", poly2_kernel),
        ("rbf", rbf_kernel),
    ]

    if QPSOLVERS_AVAILABLE:
        kernel_qp_results = {}
        for kernel_name, kernel_func in kernel_qp_list:
            for C in [0.1, 1, 10]:
                alpha, b_value, support_mask = solve_kernel_svm_qp(
                    X_13_qp,
                    y_13_qp,
                    kernel_func,
                    C=C,
                )

                model = QPKernelSVM(X_13_qp, y_13_qp, alpha, b_value, kernel_func)

                err_sub = error_probability(model, X_13_qp, y_13_qp)
                err_full = error_probability(model, X_13, y_13)

                print(f"solve_qp, kernel={kernel_name}, C={C}")
                print("w_N =", b_value)
                print("Количество опорных векторов на QP-подвыборке =", np.sum(support_mask))
                print("Ошибка на QP-подвыборке p =", err_sub)
                print("Ошибка на полной выборке p =", err_full)
                print()

                plot_decision_function(
                    model,
                    X1,
                    X3,
                    f"solve_qp: {kernel_name}, C={C}",
                    OUT / f"lab4_kernel_qp_{kernel_name}_C_{str(C).replace('.', '_')}.png",
                )

                kernel_qp_results[(kernel_name, C)] = {
                    "b": float(b_value),
                    "support_vectors": int(np.sum(support_mask)),
                    "error_sub": float(err_sub),
                    "error_full": float(err_full),
                }

        results["kernel_qp"] = kernel_qp_results
    else:
        print("Пропущено: qpsolvers не установлен.")

    # ------------------------------------------
    # Ядерный SVM через SVC
    # ------------------------------------------
    print("\n7) Ядерный SVM через sklearn.svm.SVC")
    kernel_sklearn_list = [
        ("poly", {"degree": 2}),
        ("rbf", {"gamma": 1}),
        ("sigmoid", {"gamma": 0.1, "coef0": -1}),
    ]

    kernel_svc_results = {}
    for kernel_name, params in kernel_sklearn_list:
        for C in [0.1, 1, 10]:
            model = SVC(kernel=kernel_name, C=C, **params)
            model.fit(X_13, y_13)

            print_model_info(
                model,
                X_13,
                y_13,
                f"SVC, kernel={kernel_name}, C={C}",
            )

            plot_decision_function(
                model,
                X1,
                X3,
                f"SVC: kernel={kernel_name}, C={C}",
                OUT / f"lab4_kernel_svc_{kernel_name}_C_{str(C).replace('.', '_')}.png",
            )

            kernel_svc_results[(kernel_name, C)] = {
                "error": float(error_probability(model, X_13, y_13)),
                "support_vectors": int(len(model.support_vectors_)),
            }

    results["kernel_svc"] = kernel_svc_results

    # ------------------------------------------
    # Сводная таблица ошибок
    # ------------------------------------------
    summary_models = [
        ("SVC linear C=0.1", SVC(kernel="linear", C=0.1)),
        ("SVC linear C=1", SVC(kernel="linear", C=1)),
        ("SVC linear C=10", SVC(kernel="linear", C=10)),
        (f"SVC linear best C={best_C}", SVC(kernel="linear", C=best_C)),
        ("SVC poly degree=2 C=1", SVC(kernel="poly", degree=2, C=1)),
        ("SVC rbf gamma=1 C=1", SVC(kernel="rbf", gamma=1, C=1)),
        ("SVC sigmoid C=1", SVC(kernel="sigmoid", gamma=0.1, coef0=-1, C=1)),
    ]

    print("\n8) Сводка по SVC-моделям")
    summary_lines = []
    for name, model in summary_models:
        model.fit(X_13, y_13)
        p = error_probability(model, X_13, y_13)
        print(name, "p =", p)
        summary_lines.append(f"{name}: p={p:.6f}")

    # -----------------------------------------------------------------
    # Запись отчета
    # -----------------------------------------------------------------
    lines = []
    lines.append("Лабораторная 4")
    lines.append("=" * 60)
    lines.append("")
    lines.append("1) Линейно разделимые выборки")
    lines.append(f"Bayes error = {results['separable']['bayes_error']:.6f}")
    lines.append(f"SVC error = {results['separable']['svc_error']:.6f}")
    lines.append(f"LinearSVC error = {results['separable']['linearsvc_error']:.6f}")
    if results["separable"]["qp"] is not None:
        lines.append(f"QP error = {results['separable']['qp']['error']:.6f}")
        lines.append(f"QP support vectors = {results['separable']['qp']['support_vectors']}")
    lines.append("")
    lines.append("2) Линейно неразделимые классы 1 и 3")
    lines.append(f"Bayes error = {results['neq_linear']['bayes_error']:.6f}")
    lines.append(f"Best C = {results['neq_linear']['best_C']}")
    lines.append(f"Best SVC error = {results['neq_linear']['best_error']:.6f}")
    if "neq_qp" in results:
        lines.append("QP results:")
        for C, info in results["neq_qp"].items():
            lines.append(
                f"  C={C}: error={info['error_full']:.6f}, support_vectors={info['support_vectors']}"
            )
    lines.append("")
    lines.append("3) Ядерный SVM через solve_qp")
    if "kernel_qp" in results:
        for (kernel_name, C), info in results["kernel_qp"].items():
            lines.append(
                f"  {kernel_name}, C={C}: error_full={info['error_full']:.6f}, support_vectors={info['support_vectors']}"
            )
    lines.append("")
    lines.append("4) Ядерный SVM через SVC")
    for (kernel_name, C), info in results["kernel_svc"].items():
        lines.append(
            f"  {kernel_name}, C={C}: error={info['error']:.6f}, support_vectors={info['support_vectors']}"
        )
    lines.append("")
    lines.append("5) Сводка")
    lines.extend(summary_lines)
    lines.append("")
    lines.append("Файлы графиков:")
    for name in sorted(p.name for p in OUT.glob("lab4*.png")):
        lines.append(name)

    (OUT / "results.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nГотово. Результаты сохранены в: {OUT}")


def main():
    run_lab4()


if __name__ == "__main__":
    main()

# =============================================================================
# Аналитическая часть (кратко)
#
# Лабораторная 4.
# Линейный SVM:
#   w^T w -> min
#   r_j (w^T x_j + w_0) >= 1
#
# Двойственная задача:
#   - sum lambda_j + 1/2 sum sum lambda_i lambda_j r_i r_j <x_i, x_j> -> min
#   sum lambda_j r_j = 0
#   lambda_j >= 0
#
# Мягкий зазор:
#   1/2 w^T w + C sum s_j -> min
#   r_j (w^T x_j + w_0) >= 1 - s_j
#   s_j >= 0
#   0 <= lambda_j <= C
#
# Ядра:
#   K(x, y) = x^T y
#   K(x, y) = (x^T y + 1)^2
#   K(x, y) = exp(-gamma ||x-y||^2)
#   K(x, y) = tanh(gamma x^T y + c)
#
# Экспериментальная вероятность ошибки:
#   p = 1 - Accuracy
# =============================================================================
