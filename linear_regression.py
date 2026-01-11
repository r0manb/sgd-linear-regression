from typing import Generator

import numpy as np
from numpy.typing import ArrayLike, NDArray

_rng = np.random.default_rng()


class SGDLinearRegression:
    _batch_size: int | None
    _lr: float
    _shuffle: bool
    _epochs: int
    _rng: Generator
    _weights: NDArray | None

    def __init__(
        self,
        epochs: int = 100,
        batch_size: int | None = None,
        lr: float = 0.001,
        shuffle: bool = True,
        random_state: int | None = None,
    ) -> None:
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = lr
        self._shuffle = shuffle
        self._weights = None
        self._rng = (
            _rng if random_state is None else np.random.default_rng(seed=random_state)
        )

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        X = self._add_bias_feature(np.asarray(X))
        y = np.asarray(y)

        if (self._weights is None) or (self._weights.shape[0] != X.shape[1]):
            self._weights = self._rng.random(size=X.shape[1])

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

                preds = self.predict(X_batch)
                errors = preds - y_batch
                grad = (2 / len(X_batch)) * (X_batch.T @ errors)

                self._weights -= self._lr * grad

    def predict(self, X: ArrayLike) -> NDArray:
        if self._weights is None:
            raise ValueError("Модель не была обучена. Для начала вызовите fit()")

        X = self._add_bias_feature(np.asarray(X))
        return X @ self._weights

    def _add_bias_feature(self, X: NDArray) -> NDArray:
        feature = np.ones(X.shape[0])
        return np.insert(X, 0, feature, axis=1)

    @property
    def weights(self) -> NDArray:
        return self._weights.copy()
