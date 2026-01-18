import numpy as np
from numpy.typing import ArrayLike, NDArray

from linear_regression import SGDLinearRegression


class SGDLogisticRegression(SGDLinearRegression):
    def _base_loss(self, X: NDArray, y: NDArray) -> float:
        probs = self._predict(X)
        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))

    def _base_loss_grad(self, X: NDArray, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return (1 / len(X)) * X.T @ (y_pred - y_true)

    def _predict(self, X: NDArray) -> NDArray:
        return 1 / (1 + np.exp(-super()._predict(X)))

    def predict(self, X: ArrayLike, threshold: float = 0.5) -> NDArray:
        return self.predict_proba(X) >= threshold

    def predict_proba(self, X: ArrayLike) -> NDArray:
        return super().predict(X)

    def _log_epoch(self, epoch: int, loss: float, base_loss: float) -> None:
        print(
            f"Epoch {epoch + 1}/{self._epochs} - loss: {loss:.6f}, "
            f"log-loss: {base_loss:.6f}"
        )
