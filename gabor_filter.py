import numpy as np
from numpy import cos, sin, pi, log, exp, ceil, sqrt, abs
from scipy.signal import convolve2d
from scipy.misc.pilutil import toimage
import PIL.Image as Image


class GaborFilter(object):

    def __init__(self, wavelength, theta, gamma=2, a=0.5, constrain=False):

        print("create instance")

        # Compute appropriate pixel dimensions for longest axis
        eta = -0.5 / (a * wavelength) ** 2
        if(constrain):
            size = wavelength
        else:
            size = 1 + 2 * \
                int(ceil(sqrt(log(0.05) * max(1, gamma ** 2) / eta)))

        # Initialize mask dimensions
        self.mask = np.zeros((size, size), 'float32')

        # Generate the mask for the filter
        for x in range(size):
            for y in range(size):
                tx = x - (size - 1) / 2.0
                ty = y - (size - 1) / 2.0
                rx = tx * cos(theta) + ty * sin(theta)
                ry = -1 * tx * sin(theta) + ty * cos(theta)
                ex = exp(eta * (rx ** 2 + ry ** 2 / gamma ** 2))
                self.mask[x, y] = ex * sin(2 * pi * rx / wavelength)

        # Normalize the filter with average 0 and magnitude 1
        self.mask -= self.mask.sum() / self.mask.size
        self.mask /= sqrt((self.mask * self.mask).sum())

    def convolve(self, image):
        return abs(convolve2d(image, self.mask, mode='same'))

    def show(self):
        #        toimage(self.mask).show()
        toimage(self.mask).save("./wave.png")
        print("wave show call")


if __name__ == '__main__':
    gf = GaborFilter(10, 0)

    pic = Image.open("./srcimg50.png")
    pix = np.asarray(pic)

    convarr = gf.convolve(pix)
    convimg = Image.fromarray(convarr)
    convimg.convert('RGB').save('convimg.png')

    print(convimg)
    gf.show()
    print("end")
