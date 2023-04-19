import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from cs231n.data_utils import load_CIFAR10


def linear_func(W, x):
    """linear func"""
    return np.dot(x, W)


def get_mini_batch(X, y, batch_size):
    """add random and avoid too much
    data in one epoch/step"""
    m, n = X.shape
    indices = np.random.choice(m, batch_size)
    return X[indices], y[indices]


def unify(x):
    """resize to 32x32x3 and unify
    value of x to 0~255"""
    print(x.shape)
    x = x.reshape(32, 32, 3)
    min = np.min(x)
    max = np.max(x)
    x = (x - min) / (max - min) * 255
    x = x.astype(np.uint8)
    return x


def forward_backward(W, X, y, reg):
    """
    - W: n' x n
    - X: m x n'
    - y: m
    """
    m = X.shape[0]
    n = W.shape[1]

    scores = np.dot(X, W)  # m x n
    score = np.choose(y, np.transpose(scores))  # m
    compares = 1 + scores - score[:, None]  # m x n

    margins = np.maximum(0, compares)  # m x n
    loss = (np.sum(margins) - m) / m + reg * np.sum(np.square(W))

    masks1 = compares > 0  # m x n
    mask1 = np.sum(masks1, axis=1)  # m x 1
    masks2 = np.tile(np.arange(n), [m, 1]) == y[:, None]  # m x n
    dW = np.dot(np.transpose(X), masks1 - mask1[:, None] * masks2) / m + 2 * reg * W

    return loss, dW


class SVM:

    def __init__(self, k, step=10000, learning_rate=0.01, batch_size=256, reg=0.000005):
        self.k = k
        self.step = step
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.reg = reg
        self.W = None

    def fit(self, X, y):
        m, n = X.shape
        X = np.column_stack((np.ones(m), X))
        m, n = X.shape
        y -= 1

        self.W = np.zeros((n, self.k))
        for _ in range(self.step):
            X_, y_ = get_mini_batch(X, y, self.batch_size)
            loss, dW = forward_backward(self.W, X_, y_, self.reg)
            print("train step is %d, loss is %0.2f%%" % (_, loss * 100))
            self.W -= self.learning_rate * dW

    def predict(self, X):
        m, n = X.shape
        X = np.column_stack((np.ones(m), X))
        return np.array([np.argmax(linear_func(self.W, x)) for x in X]) + 1


def show_iris():
    # step1: classifier
    clf = SVM(2)

    # step2: fit
    iris = datasets.load_iris()
    train_x = iris.data[:100, :2]
    train_y = iris.target[:100] + 1
    clf.fit(train_x, train_y)

    # step3: predict
    y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 1
    x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    ZZ = np.array(Z).reshape(xx.shape)

    # step4: plot
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    plt.figure()
    plt.pcolormesh(xx, yy, ZZ, cmap=cmap_light)
    plt.scatter(train_x[:, 0], train_x[:, 1], c=train_y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


def show_cifar():
    Xtr, Ytr, Xte, Yte = load_CIFAR10('cs231n/datasets/cifar-10-batches-py/')
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)
    Ytr += 1
    Yte += 1

    svm = SVM(10, step=5000)
    svm.fit(Xtr_rows, Ytr)

    print(np.mean(svm.predict(Xte_rows) == Yte))


def show_w():
    Xtr, Ytr, Xte, Yte = load_CIFAR10('cs231n/datasets/cifar-10-batches-py/')
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)
    Ytr += 1
    Yte += 1

    svm = SVM(10, step=5000)
    svm.fit(Xtr_rows, Ytr)

    fig, axes = plt.subplots(2, 5)
    for i in range(10):
        w = svm.W[:, i]
        w_ = w[1:]
        w_ = unify(w_)
        x, y = i // 5, i % 5
        axes[x][y].imshow(w_)
        axes[x][y].set_xticks([])
        axes[x][y].set_yticks([])
    plt.show()


def show_pic():
    Xtr, Ytr, Xte, Yte = load_CIFAR10('cs231n/datasets/cifar-10-batches-py/')
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)
    Ytr += 1
    Yte += 1

    svm = SVM(10, step=5000)
    svm.fit(Xtr_rows, Ytr)

    t, k = 10, 10
    fig, axes = plt.subplots(t, k + 2)
    for index_x in range(t):
        x = Xte_rows[index_x]
        axes[index_x][0].imshow(x.reshape(32, 32, 3).astype(np.uint8))
        axes[index_x][0].set_xticks([])
        axes[index_x][0].set_yticks([])
        axes[index_x][1].set_xticks([])
        axes[index_x][1].set_yticks([])

        for index_y in range(k):
            w = svm.W[:, index_y]
            x_ = w[1:] * x
            x_ = unify(x_)
            axes[index_x][index_y + 2].imshow(x_)
            axes[index_x][index_y + 2].set_xticks([])
            axes[index_x][index_y + 2].set_yticks([])
    plt.show()


if __name__ == "__main__":
    # show_iris()
    # show_cifar()
    # show_w()
    show_pic()
