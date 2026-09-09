from math import log,exp
from re import L


type Vector = list[float]
type Matrix = list[Vector]

def _dot(a: Vector,b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("Error: size matrix a != size matrix b")

    ans = 0.0

    for i in range(len(a)):
        ans += a[i] * b[i]

    return ans

def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + exp(-value))

    return 1.0 / (1.0 + exp(value))


class LogistRegression:
    def __init__(
        self,
        learning_rate: float = 0.0,
        threshold: float = 0.5,
    ) -> None:
        self.weigths: Vector = []
        self.bias: float = 0.0
        self.learning_rate = learning_rate
        self.threshold = threshold

    def _predict_one(self,sample: Vector) -> float:
        z = _dot(self.weigths, sample)
        return _sigmoid(z)

    def predict_proba(self, x: Matrix) -> Vector:
        probas: Vector = []

        for sample in x:
            probas.append(self._predict_one(sample))

        return probas

    def predict(self, x: Matrix) ->list[int]:
        proba = self.predict_proba(x)
        predictions: list[int] = []

        for p in proba:
            if p >= self.threshold:
                predictions.append(1)
            else:
                predictions.append(0)

        return predictions

    def _binary_cross_entropy(
        self,
        x: Matrix,
        y: list[int],
    ) -> float:
        epsilon = 1e-15
        total_loss = 0.0

        for i in range(len(x)):
            proba = self._predict_one(x[i])

            proba = max(
                epsilon,
                min(1.0 - epsilon, proba)
            )

            total_loss -= y[i] * log(proba) + (1 - y[i]) * log(1.0 - proba)

        return total_loss / len(x)

    def _gradient(
        self,
        x: Matrix,
        y: list[int],
    ) -> tuple[Vector, float]:
        weight_gra = [0.0] * len(self.weigths)
        bias_grad = 0.0

        for i in range(len(x)):
            probability = self._predict_one(x[i])
            error = probability - y[i]

            for j in range(len(self.weigths)):
                weight_gra[j] += error * x[i][j]

            bias_grad += error

        for j in range(len(weight_gra)):
            weight_gra[j] /= len(x)

        bias_grad /= len(x)

        return weight_gra, bias_grad

    def train(
        self,
        x: Matrix,
        y: list[int],
        epochs: int = 1_000
    ) -> None:
        feature_cnt = len(x[0])
        self.weigths = [0.0] * feature_cnt
        self.bias = 0.0

        for _ in range(epochs):
            gradient_w, gradient_b = self._gradient(x,y)

            for j in range(len(self.weigths)):
                self.weigths[j] -= self.learning_rate * gradient_w[j]

            self.bias -= self.learning_rate * gradient_b
