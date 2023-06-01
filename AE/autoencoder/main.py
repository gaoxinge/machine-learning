import tensorflow as tf
import matplotlib.pyplot as plt


class AE(tf.keras.models.Model):

    def __init__(self, latent_dim):
        super(AE, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim, activation="relu"),
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(784, activation="sigmoid"),
            tf.keras.layers.Reshape((28, 28))
        ])

    def call(self, x, training=None, mask=None):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


(x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

latent_dim = 64
ae = AE(latent_dim)
ae.compile(optimizer="adam", loss=tf.losses.MeanSquaredError())
ae.fit(x_train, x_train, epochs=20, validation_data=(x_test, x_test))

x = tf.random.normal([1, 64], 0, 1, tf.float32)
x = ae.decoder(x)
x = x.numpy()
plt.imshow(x[0])
plt.gray()
plt.show()

