import numpy as np
import csv
from keras.datasets import mnist
from vae import VAE
from time import time

def score_parameters(x_train, x_test, original_dim, **kwargs):
    """ Train the auto encoder and score. """
    factory = VAE(original_dim=original_dim, **kwargs)
    vae = factory.compile()
    vae.fit(x_train,
           epochs=10,
           batch_size=30,
           validation_data=(x_test, None),
           verbose=False)
    return factory, vae, vae.history.history["val_loss"][-1]


# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Original image dimension - need to reconstruct when plotting.
image_dim = (x_train.shape[1], x_train.shape[2])
original_dim = x_train.shape[1] * x_train.shape[2]

x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Score points
x = np.linspace(1, 1000, 1000).astype(int)
np.random.shuffle(x)
fp = open("out.txt", "w", buffering=1)
writer = csv.DictWriter(fp, fieldnames=["x", "score", "time"])
writer.writeheader()

# Write scores to file
for xi in x:
    t = time()
    score = score_parameters(x_train, x_test, original_dim, intermediate_dim=xi)[-1]
    t = time() - t
    writer.writerow({"x": xi, "score": score, "time": t})
    print("hook\tx: %d\tscore: %.3f\ttime: %.2f" % (xi, score, t))
fp.close()