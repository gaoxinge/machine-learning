import numpy as np
import matplotlib.pyplot as plt
from cs231n.data_utils import load_CIFAR10


class NearestNeighbor:

    def __init__(self, k):
        self.k = k

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, x):
        index = self.topK(x)
        bs, cs = np.unique(index, return_counts=True)
        return self.ytr[bs[np.argmax(cs)]]

    def topK(self, x):
        distances = np.sum(np.square(self.Xtr - x), axis=1)
        return np.argpartition(distances, self.k)[:self.k]


if __name__ == "__main__":
    Xtr, Ytr, Xte, Yte = load_CIFAR10('cs231n/datasets/cifar-10-batches-py/')
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)

    t, k = 10, 10
    nn = NearestNeighbor(k)
    nn.train(Xtr_rows, Ytr)

    # cnt0, cnt1 = 0, 0
    # for index_x in range(Xte.shape[0]):
    #     x = Xte_rows[index_x]
    #     y0 = nn.predict(x)
    #     y1 = Yte[index_x]
    #     if y0 == y1:
    #         cnt0 += 1
    #     cnt1 += 1
    #     if cnt1 % 10 == 0:
    #         print("progress: %0.2f%%, accuracy: %0.2f%%" % (cnt1 / Xte.shape[0] * 100, cnt0 / cnt1 * 100))
    # print("accuracy: %0.2f%%" % (cnt0 / cnt1 * 100,))

    fig, axes = plt.subplots(t, k + 2)
    for index_x in range(t):
        x = Xte_rows[index_x]
        axes[index_x][0].imshow(x.reshape(32, 32, 3).astype(np.uint8))
        axes[index_x][0].set_xticks([])
        axes[index_x][0].set_yticks([])
        axes[index_x][1].set_xticks([])
        axes[index_x][1].set_yticks([])

        y_top_k = Xtr_rows[nn.topK(x)]
        for index_y in range(k):
            y_top = y_top_k[index_y]
            axes[index_x][index_y + 2].imshow(y_top.reshape(32, 32, 3).astype(np.uint8))
            axes[index_x][index_y + 2].set_xticks([])
            axes[index_x][index_y + 2].set_yticks([])
    plt.show()
