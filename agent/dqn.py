import numpy as np
"""
DQN на чистом numpy
Вместо CNN используем полносвязную сеть (MLP) — проще для понимания,
и достаточно для нашей маленькой задачи (84x84 = 7056 входов, 12 выходов).

Архитектура:
  Вход: 84*84 = 7056 пикселей (один кадр, flattened)
  Слой 1: 7056 -> 128 нейронов + ReLU
  Слой 2: 128 -> 64 нейрона + ReLU
  Выход:  64 -> 12 (Q-значение для каждого действия)
"""


class DQN:
    def __init__(self, input_size=900, hidden1=32, hidden2=24, num_actions=12, lr=1e-4):
        self.lr = lr

        # Инициализация весов (He initialization — работает лучше чем случайная)
        # Каждый слой: weights (матрица) + biases (вектор)
        scale1 = np.sqrt(2.0 / input_size)
        self.w1 = np.random.randn(input_size, hidden1).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden1, dtype=np.float32)

        scale2 = np.sqrt(2.0 / hidden1)
        self.w2 = np.random.randn(hidden1, hidden2).astype(np.float32) * scale2
        self.b2 = np.zeros(hidden2, dtype=np.float32)

        scale3 = np.sqrt(2.0 / hidden2)
        self.w3 = np.random.randn(hidden2, num_actions).astype(np.float32) * scale3
        self.b3 = np.zeros(num_actions, dtype=np.float32)

    def forward(self, x):
        """
        Прямой проход: вход -> Q-значения.
        x: (batch, input_size)
        return: (batch, num_actions)

        Сохраняем промежуточные значения — они нужны для backprop.
        """
        # Слой 1: linear + ReLU
        self.z1 = x @ self.w1 + self.b1           # (batch, 128)
        self.a1 = np.maximum(0, self.z1)           # ReLU: если < 0 → 0

        # Слой 2: linear + ReLU
        self.z2 = self.a1 @ self.w2 + self.b2     # (batch, 64)
        self.a2 = np.maximum(0, self.z2)           # ReLU

        # Выходной слой: linear (без активации — Q-значения могут быть любыми)
        self.out = self.a2 @ self.w3 + self.b3    # (batch, 12)

        # Сохраняем вход для backprop
        self.x = x
        return self.out

    def backward(self, dout):
        """
        Обратный проход: считаем градиенты и обновляем веса.
        dout: (batch, num_actions) — градиент ошибки по выходу

        Это цепное правило (chain rule) в действии:
        "∂Loss/∂w = ∂Loss/∂out × ∂out/∂hidden × ∂hidden/∂w"
        """
        batch_size = dout.shape[0]

        # Градиенты выходного слоя
        dw3 = self.a2.T @ dout / batch_size
        db3 = np.mean(dout, axis=0)
        dhidden2 = dout @ self.w3.T

        # Градиенты через ReLU слоя 2
        # ReLU: градиент = 1 если вход > 0, иначе 0
        dhidden2 = dhidden2 * (self.z2 > 0).astype(np.float32)

        # Градиенты слоя 2
        dw2 = self.a1.T @ dhidden2 / batch_size
        db2 = np.mean(dhidden2, axis=0)
        dhidden1 = dhidden2 @ self.w2.T

        # Градиенты через ReLU слоя 1
        dhidden1 = dhidden1 * (self.z1 > 0).astype(np.float32)

        # Градиенты слоя 1
        dw1 = self.x.T @ dhidden1 / batch_size
        db1 = np.mean(dhidden1, axis=0)

        # Clip gradients (защита от взрыва)
        for g in [dw1, db1, dw2, db2, dw3, db3]:
            np.clip(g, -1.0, 1.0, out=g)

        # Обновление весов (gradient descent)
        self.w3 = (self.w3 - self.lr * dw3).astype(np.float32)
        self.b3 = (self.b3 - self.lr * db3).astype(np.float32)
        self.w2 = (self.w2 - self.lr * dw2).astype(np.float32)
        self.b2 = (self.b2 - self.lr * db2).astype(np.float32)
        self.w1 = (self.w1 - self.lr * dw1).astype(np.float32)
        self.b1 = (self.b1 - self.lr * db1).astype(np.float32)

    def copy_weights_from(self, other):
        """Копировать веса из другой сети (для target network)."""
        self.w1 = other.w1.copy()
        self.b1 = other.b1.copy()
        self.w2 = other.w2.copy()
        self.b2 = other.b2.copy()
        self.w3 = other.w3.copy()
        self.b3 = other.b3.copy()

    def save(self, path):
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2, w3=self.w3, b3=self.b3)

    def load(self, path):
        data = np.load(path)
        self.w1, self.b1 = data['w1'], data['b1']
        self.w2, self.b2 = data['w2'], data['b2']
        self.w3, self.b3 = data['w3'], data['b3']
