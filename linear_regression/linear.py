type Vector = list[float]
type Matrix = list[Vector]


def _dot(a: Vector, b: Vector) -> float:
        if len(a) != len(b):
            raise ValueError("Error: vectors must have same size")

        res = 0.0

        for i in range(len(a)):
            res += a[i] * b[i]

        return res

class LinearRegression:
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.weights: Vector = []
        self.bias = 0.0
        self.learning_rate = learning_rate

    def _validate_data(self, x: Matrix, y: Vector) -> None:
        if len(x) == 0:
            raise ValueError("x must not be empty")

        if len(x) != len(y):
            raise ValueError("x and y must contain the same number of samples")

        feature_count = len(x[0])

        if feature_count == 0:
            raise ValueError("samples must contain at least one feature")

        for sample in x:
            if len(sample) != feature_count:
                raise ValueError("all samples must have the same number of features")

    def _predict_one(self, x: Vector) -> float:
        return _dot(self.weights, x) + self.bias

    def predict(self, x: Matrix) -> Vector:
        prediction: Vector = []

        for sam in x:
            prediction.append(self._predict_one(sam))

        return prediction

    def _mse(self, x: Matrix, y: Vector) -> float:
        self._validate_data(x, y)

        score = 0.0

        for i in range(len(x)):
            prediction = self._predict_one(x[i])
            error = prediction - y[i]
            score += error**2

        return score / len(x)

    def mse(self, x: Matrix, y: Vector) -> float:
        return self._mse(x, y)

    def _gradient_weights(self, x: Matrix, y: Vector) -> Vector:
        self._validate_data(x, y)

        grad: Vector = [0.0] * len(self.weights)

        for i in range(len(x)):
            prediction = self._predict_one(x[i])
            error = prediction - y[i]

            for j in range(len(self.weights)):
                grad[j] += error * x[i][j]

        for j in range(len(grad)):
            grad[j] *= 2.0 / len(x)

        return grad

    def gradient_w(self, x: Matrix, y: Vector) -> Vector:
        return self._gradient_weights(x, y)

    def _gradient_bias(self, x: Matrix, y: Vector) -> float:
        self._validate_data(x, y)

        gradient_sum = 0.0

        for i in range(len(x)):
            pred = self._predict_one(x[i])
            error = pred - y[i]
            gradient_sum += error

        return 2.0 * gradient_sum / len(x)

    def gradient_b(self, x: Matrix, y: Vector) -> float:
        return self._gradient_bias(x, y)

    def train(self, x: Matrix, y: Vector, epochs: int = 1_000) -> None:
        self._validate_data(x, y)

        feature_count = len(x[0])
        self.weights = [0.0] * feature_count
        self.bias = 0.0

        for _ in range(epochs):
            gradient_w = self._gradient_weights(x, y)
            gradient_b = self._gradient_bias(x, y)

            for j in range(len(self.weights)):
                self.weights[j] -= self.learning_rate * gradient_w[j]

            self.bias -= self.learning_rate * gradient_b
