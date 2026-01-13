from typing import Generator, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

_rng = np.random.default_rng()


class SGDLinearRegression:
    _epochs: int
    _batch_size: int | None
    _lr: float
    _l1_ratio: float
    _alpha: float
    _shuffle: bool
    _verbose: bool
    _weights: NDArray | None
    _loss_history: list[float]
    _mse_history: list[float]
    _rng: Generator

    def __init__(
        self,
        epochs: int = 100,
        batch_size: int | None = None,
        lr: float = 0.001,
        penalty: Literal["l1", "l2", "elasticnet"] | None = None,
        alpha: float = 0.0,
        l1_ratio: float = 0.0,
        shuffle: bool = True,
        random_state: int | None = None,
        verbose: bool = True,
    ) -> None:
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = self._init_lr(lr)
        self._alpha, self._l1_ratio = self._init_ratios(penalty, alpha, l1_ratio)
        self._shuffle = shuffle
        self._verbose = verbose

        self._weights = None
        self._loss_history = []
        self._mse_history = []
        self._rng = (
            _rng if random_state is None else np.random.default_rng(seed=random_state)
        )

    def _init_lr(self, lr: float) -> float:
        if lr <= 0:
            raise ValueError("Параметр lr должен быть > 0!")

        return lr

    def _init_ratios(
        self, penalty: str | None, alpha: float, l1_ratio: float
    ) -> tuple[float, float]:
        if alpha < 0:
            raise ValueError("Параметр alpha должен быть >= 0!")
        if not (0 <= l1_ratio <= 1):
            raise ValueError("Параметр l1_ratio должен быть в отрезке [0, 1]!")

        if penalty is None:
            alpha = 0.0
            l1_ratio = 0.0
        elif penalty == "l1":
            l1_ratio = 1.0
        elif penalty == "l2":
            l1_ratio = 0.0
        elif penalty != "elasticnet":
            raise ValueError("Неизвестный регуляризатор!")

        return alpha, l1_ratio

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        X = self._add_bias_feature(np.asarray(X))
        y = np.asarray(y)

        if (self._weights is None) or (self._weights.shape[0] != X.shape[1]):
            self._weights = self._rng.normal(scale=0.01, size=X.shape[1])

        n = X.shape[0]
        batch_size = n if self._batch_size is None else self._batch_size
        idxs = np.arange(n)
        for epoch in range(self._epochs):
            if self._shuffle:
                self._rng.shuffle(idxs)

            for start in range(0, n, batch_size):
                batch_idxs = idxs[start : min(start + batch_size, n)]
                X_batch = X[batch_idxs]
                y_batch = y[batch_idxs]

                preds = self._predict(X_batch)
                grad = self._loss_grad(X_batch, y_batch, preds)

                self._weights -= self._lr * grad

            loss = self._loss(X, y)
            mse = self._mse(X, y)
            self._loss_history.append(loss)
            self._mse_history.append(mse)
            if self._verbose:
                print(
                    f"Epoch {epoch + 1}/{self._epochs} - loss: {loss:.6f}, mse: {mse}"
                )

    def predict(self, X: ArrayLike) -> NDArray:
        self._check_fitted()

        X = self._add_bias_feature(np.asarray(X))
        return self._predict(X)

    def _loss_grad(self, X: NDArray, y_true: NDArray, y_pred: NDArray) -> NDArray:
        grad = self._mse_grad(X, y_true, y_pred)
        if self._alpha > 0:
            grad += self._reg_grad()

        return grad

    def _mse_grad(self, X: NDArray, y_true: NDArray, y_pred: NDArray) -> NDArray:
        errors = y_pred - y_true
        grad = (2 / len(X)) * (X.T @ errors)
        return grad

    def _reg_grad(self) -> NDArray:
        w = self._weights[1:]
        grad = np.zeros_like(self._weights)

        grad[1:] = self._alpha * (
            self._l1_ratio * np.size(w) + (1 - self._l1_ratio) * 2 * w
        )

        return grad

    def _loss(self, X: NDArray, y: NDArray) -> float:
        loss = self._mse(X, y)
        if self._alpha > 0:
            loss += self._reg()

        return loss

    def _mse(self, X: NDArray, y: NDArray) -> float:
        return np.mean((self._predict(X) - y) ** 2)

    def _reg(self) -> float:
        w = self._weights[1:]
        l1_reg = self._l1_ratio * np.abs(w).sum()
        l2_reg = (1 - self._l1_ratio) * (w**2).sum()

        return self._alpha * (l1_reg + l2_reg)

    def _predict(self, X: NDArray) -> NDArray:
        return X @ self._weights

    def _add_bias_feature(self, X: NDArray) -> NDArray:
        feature = np.ones(X.shape[0])
        return np.insert(X, 0, feature, axis=1)

    @property
    def weights(self) -> NDArray:
        self._check_fitted()

        return self._weights[1:].copy()

    @property
    def bias(self) -> float:
        self._check_fitted()

        return self._weights[0]

    @property
    def loss_history(self) -> list[float]:
        return self._loss_history.copy()

    @property
    def mse_history(self) -> list[float]:
        return self._mse_history.copy()

    def _check_fitted(self) -> None:
        if self._weights is None:
            raise ValueError("Модель не была обучена. Для начала вызовите fit()")
