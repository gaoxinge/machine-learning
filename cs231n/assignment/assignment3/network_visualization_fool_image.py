import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from cs231n.classifiers.squeezenet import SqueezeNet
from cs231n.data_utils import load_imagenet_val
from cs231n.image_utils import preprocess_image, deprocess_image


def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image, a numpy array of shape (1, 224, 224, 3)
    - target_y: An integer in the range [0, 1000)
    - model: Pretrained SqueezeNet model

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
    """

    # Make a copy of the input that we will modify
    X_fooling = X.copy()

    # Step size for the update
    learning_rate = 1

    ##############################################################################
    # TODO: Generate a fooling image X_fooling that the model will classify as   #
    # the class target_y. Use gradient *ascent* on the target class score, using #
    # the model.scores Tensor to get the class scores for the model.image.       #
    # When computing an update step, first normalize the gradient:               #
    #   dX = learning_rate * g / ||g||_2                                         #
    #                                                                            #
    # You should write a training loop, where in each iteration, you make an     #
    # update to the input image X_fooling (don't modify X). The loop should      #
    # stop when the predicted class for the input is the same as target_y.       #
    #                                                                            #
    # HINT: Use tf.GradientTape() to keep track of your gradients and            #
    # use tape.gradient to get the actual gradient with respect to X_fooling.    #
    #                                                                            #
    # HINT 2: For most examples, you should be able to generate a fooling image  #
    # in fewer than 100 iterations of gradient ascent. You can print your        #
    # progress over iterations to check your algorithm.                          #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    X_fooling = tf.Variable(X_fooling)
    Y_t = [[0 for _ in range(1000)]]
    Y_t[0][target_y] = 1
    for _ in range(100):
        with tf.GradientTape() as tape:
            tape.watch(X_fooling)
            Y = model(X_fooling)
            if tf.math.argmax(Y[0]).numpy() == target_y:
                break
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=Y_t, logits=Y)
        dX = tape.gradient(loss, X_fooling)
        dX = dX * learning_rate / tf.sqrt(tf.math.reduce_sum(dX * dX))
        X_fooling.assign_sub(dX)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_fooling


SAVE_PATH = "cs231n/datasets/squeezenet.ckpt"
if not os.path.exists(SAVE_PATH + ".index"):
    raise ValueError("You need to download SqueezeNet!")
model = SqueezeNet()
model.load_weights(SAVE_PATH)
model.trainable = False

X_raw, y, class_names = load_imagenet_val(num=5)
X = np.array([preprocess_image(img) for img in X_raw])
idx = 0
Xi = X[idx][None]
target_y = 6
X_fooling = make_fooling_image(Xi, target_y, model)

# Make sure that X_fooling is classified as y_target
scores = model(X_fooling)
assert tf.math.argmax(scores[0]).numpy() == target_y, 'The network is not fooled!'

# Show original image, fooling image, and difference
orig_img = deprocess_image(Xi[0])
fool_img = deprocess_image(X_fooling[0])
plt.figure(figsize=(12, 6))

# Rescale
plt.subplot(1, 4, 1)
plt.imshow(orig_img)
plt.axis('off')
plt.title(class_names[y[idx]])
plt.subplot(1, 4, 2)
plt.imshow(fool_img)
plt.title(class_names[target_y])
plt.axis('off')
plt.subplot(1, 4, 3)
plt.title('Difference')
plt.imshow(deprocess_image((Xi-X_fooling)[0]))
plt.axis('off')
plt.subplot(1, 4, 4)
plt.title('Magnified difference (10x)')
plt.imshow(deprocess_image(10 * (Xi-X_fooling)[0]))
plt.axis('off')
plt.gcf().tight_layout()
plt.show()

