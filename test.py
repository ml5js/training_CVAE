import numpy as np
from scipy.misc import imsave
import matplotlib.pyplot as plt
from keras.models import load_model

decoder = load_model('decoder.h5')

def construct_numvec(digit, z=None):
    out = np.zeros((1, 12))
    out[:, digit + 2] = 1.
    if z is None:
        return(out)
    else:
        for i in range(len(z)):
            out[:, i] = z[i]
        return(out)


sample_3 = construct_numvec(3)
print(sample_3)


dig = 3
sides = 8
max_z = 1.5

img_it = 0
for i in range(0, sides):
    z1 = (((i / (sides-1)) * max_z)*2) - max_z
    for j in range(0, sides):
        z2 = (((j / (sides-1)) * max_z)*2) - max_z
        z_ = [z1, z2]
        vec = construct_numvec(dig, z_)
        print(vec)
        decoded = decoder.predict(vec)
        plt.subplot(sides, sides, 1 + img_it)
        img_it += 1
        plt.imshow(decoded.reshape(28, 28), cmap=plt.cm.gray)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=.2)
plt.show()
