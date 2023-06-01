import random
import tqdm
import numpy as np


def v2m(x):
    return np.array([x]).T


def m2v(x):
    return x[:, 0]


def i2b(x):
    s = f"{x:08b}"
    return [int(c) for c in reversed(s)]


def b2i(s):
    return int("".join([str(_) for _ in reversed(s)]), 2)


def ii2v(x1, x2):
    x1 = i2b(x1)
    x2 = i2b(x2)
    return [np.array([c1, c2], dtype=np.float32) for c1, c2 in zip(x1, x2)]


def i2v(x):
    x = i2b(x)
    return [np.array([1 - c, c], dtype=np.float32) for c in x]


def softmax(x):
    t = np.exp(x)
    return t / np.sum(t)


def loss_fn(y, z):
    return -np.sum(z * np.log(y))


class RNN:

    def __init__(self, h_dim, x_dim, y_dim, h):
        # init rand instead of zero
        self.W_hh = np.random.randn(h_dim, h_dim)
        self.W_xh = np.random.randn(h_dim, x_dim)
        self.W_hy = np.random.randn(y_dim, h_dim)
        self.b_h = np.random.randn(h_dim, 1)
        self.b_y = np.random.randn(y_dim, 1)

        # cache
        self.hs = [v2m(h)]
        self.xs = []
        self.ys = []

    def reset_cache(self):
        h = self.hs[0]
        self.hs = [h]
        self.xs = []
        self.ys = []

    def _forward(self, x):
        h = self.hs[-1]
        h = np.tanh(np.dot(self.W_hh, h) + np.dot(self.W_xh, x) + self.b_h)
        y = softmax(np.dot(self.W_hy, h) + self.b_y)

        self.hs.append(h)
        self.xs.append(x)
        self.ys.append(y)

    def forward(self, xs):
        xs = [v2m(x) for x in xs]
        for x in xs:
            self._forward(x)
        ys = [m2v(y) for y in self.ys]
        return ys

    def _backward(self, t, z):
        loss = loss_fn(self.ys[t], z)
        dy = -z + self.ys[t] * np.sum(z)

        dW_hy = np.dot(dy, self.hs[t + 1].T)
        db_y = dy

        dW_hh = np.zeros_like(self.W_hh)
        dW_xh = np.zeros_like(self.W_xh)
        db_h = np.zeros_like(self.b_h)

        p = (1 - self.hs[t + 1] * self.hs[t + 1]) * np.dot(self.W_hy.T, dy)
        for k in range(t, -1, -1):
            dW_hh += np.dot(p, self.hs[k].T)
            dW_xh += np.dot(p, self.xs[k].T)
            db_h += p
            p = (1 - self.hs[k] * self.hs[k]) * np.dot(self.W_hh.T, p)
        return loss, dW_hh, dW_xh, dW_hy, db_h, db_y

    def backward(self, zs):
        zs = [v2m(z) for z in zs]

        loss = 0
        dW_hh = np.zeros_like(self.W_hh)
        dW_xh = np.zeros_like(self.W_xh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        for t in range(len(zs) - 1, -1, -1):
            z = zs[t]
            d = self._backward(t, z)
            loss += d[0]
            dW_hh += d[1]
            dW_xh += d[2]
            dW_hy += d[3]
            db_h += d[4]
            db_y += d[5]

        return loss, dW_hh, dW_xh, dW_hy, db_h, db_y


class Optimizer:

    def __init__(self, rnn, lr=0.01):
        self.rnn = rnn
        self.lr = lr

        self.cnt = 0
        self.dW_hh = np.zeros_like(rnn.W_hh)
        self.dW_xh = np.zeros_like(rnn.W_xh)
        self.dW_hy = np.zeros_like(rnn.W_hy)
        self.db_h = np.zeros_like(rnn.b_h)
        self.db_y = np.zeros_like(rnn.b_y)

    def zero(self):
        self.cnt = 0
        self.dW_hh = np.zeros_like(rnn.W_hh)
        self.dW_xh = np.zeros_like(rnn.W_xh)
        self.dW_hy = np.zeros_like(rnn.W_hy)
        self.db_h = np.zeros_like(rnn.b_h)
        self.db_y = np.zeros_like(rnn.b_y)

    def add(self, d):
        self.cnt += 1
        self.dW_hh += d[1]
        self.dW_xh += d[2]
        self.dW_hy += d[3]
        self.db_h += d[4]
        self.db_y += d[5]

    def update(self):
        dW_hh = self.dW_hh / self.cnt
        dW_xh = self.dW_xh / self.cnt
        dW_hy = self.dW_hy / self.cnt
        db_h = self.db_h / self.cnt
        db_y = self.db_y / self.cnt

        self.rnn.W_hh -= self.lr * dW_hh
        self.rnn.W_xh -= self.lr * dW_xh
        self.rnn.W_hy -= self.lr * dW_hy
        self.rnn.b_h -= self.lr * db_h
        self.rnn.b_y -= self.lr * db_y


class OptimizerV2(Optimizer):

    def update(self):
        # https://gist.github.com/karpathy/d4dee566867f8291f086#file-min-char-rnn-py-L109
        dW_hh = self.dW_hh / self.cnt
        dW_xh = self.dW_xh / self.cnt
        dW_hy = self.dW_hy / self.cnt
        db_h = self.db_h / self.cnt
        db_y = self.db_y / self.cnt

        self.rnn.W_hh -= self.lr * dW_hh / np.sqrt(dW_hh ** 2 + 1e-8)
        self.rnn.W_xh -= self.lr * dW_xh / np.sqrt(dW_xh ** 2 + 1e-8)
        self.rnn.W_hy -= self.lr * dW_hy / np.sqrt(dW_hy ** 2 + 1e-8)
        self.rnn.b_h -= self.lr * db_h / np.sqrt(db_h ** 2 + 1e-8)
        self.rnn.b_y -= self.lr * db_y / np.sqrt(db_y ** 2 + 1e-8)


def check_dim():
    rnn = RNN(10, 2, 4, np.zeros(10))

    xs = [
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
    ]
    ys = rnn.forward(xs)
    print(len(ys), ys[0].shape)

    zs = [
        np.zeros(4),
        np.zeros(4),
        np.zeros(4),
        np.zeros(4),
        np.zeros(4),
        np.zeros(4),
        np.zeros(4),
        np.zeros(4),
    ]
    d = rnn.backward(zs)
    print(len(d), d[0], d[1].shape, d[2].shape, d[3].shape, d[4].shape, d[5].shape)


def test(rnn, a, b):
    xs = ii2v(a, b)
    rnn.reset_cache()
    ys = rnn.forward(xs)
    r = [np.argmax(_) for _ in ys]
    return b2i(r)


if __name__ == "__main__":
    # check_dim()

    rnn = RNN(16, 2, 2, np.zeros(16))
    optimizer = Optimizer(rnn, lr=0.01)

    epochs = 100
    num = 128
    bs = 32

    for epoch in range(epochs):
        loss = 0
        for _ in tqdm.tqdm(range(num)):
            optimizer.zero()
            for j in range(bs):
                a = random.randint(0, 127)
                b = random.randint(0, 127)
                c = a + b
                xs = ii2v(a, b)
                zs = i2v(c)

                rnn.reset_cache()
                ys = rnn.forward(xs)
                d = rnn.backward(zs)

                loss += d[0]
                optimizer.add(d)
            optimizer.update()
        loss /= num * bs
        print(epoch + 1, loss)

    print(f"5 + 8 = {test(rnn, 5, 8)}")

