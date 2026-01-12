from typing import Generator

import numpy as np
from numpy.typing import ArrayLike, NDArray

_rng = np.random.default_rng()


class SGDLinearRegression:
    _epochs: int
    _batch_size: int | None
    _lr: float
    _shuffle: bool
    _verbose: bool
    _weights: NDArray | None
    _loss_history: list[float]
    _rng: Generator

    def __init__(
        self,
        epochs: int = 100,
        batch_size: int | None = None,
        lr: float = 0.001,
        shuffle: bool = True,
        random_state: int | None = None,
        verbose: bool = True,
    ) -> None:
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = lr
        self._shuffle = shuffle
        self._verbose = verbose

        self._weights = None
        self._loss_history = []
        self._rng = (
            _rng if random_state is None else np.random.default_rng(seed=random_state)
        )

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
                grad = self._grad(X_batch, y_batch, preds)
                self._weights -= self._lr * grad

            loss = self._loss(X, y)
            self._loss_history.append(loss)
            if self._verbose:
                print(f"Epoch {epoch + 1}/{self._epochs} - loss: {loss:.6f}")

    def _grad(self, X: NDArray, y_true: NDArray, y_pred: NDArray) -> NDArray:
        errors = y_pred - y_true
        return (2 / len(X)) * (X.T @ errors)

    def _loss(self, X: NDArray, y: NDArray) -> float:
        return np.mean((self._predict(X) - y) ** 2)

    def predict(self, X: ArrayLike) -> NDArray:
        self._check_fitted()

        X = self._add_bias_feature(np.asarray(X))
        return self._predict(X)

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

    def _check_fitted(self) -> None:
        if self._weights is None:
            raise ValueError("Модель не была обучена. Для начала вызовите fit()")
