import numpy as np
import csv
from keras.datasets import mnist
from vae import VAE
from time import time
import sys

def score_parameters(x_train, x_test, original_dim, **kwargs):
    """ Train the auto encoder and score. """
    factory = VAE(original_dim=original_dim, latent_dim=1, **kwargs)
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
f_out = sys.argv[1]
a = [0] + list(np.logspace(-10, 0, 31))
fp = open(f_out, "w", buffering=1)
writer = csv.DictWriter(fp, fieldnames=["alpha", "score", "time"])
writer.writeheader()

# Write scores to file
for alpha in a:
    t = time()
    score = score_parameters(x_train, x_test, original_dim, intermediate_dim=512, alpha1=alpha)[-1]
    t = time() - t
    writer.writerow({"alpha": alpha, "score": score, "time": t})
    print("hook\talpha: %d\tscore: %.3f\ttime: %.2f" % (alpha, score, t))
fp.close()