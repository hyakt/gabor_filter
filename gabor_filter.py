#
# @module: And-Or Images
# @copyright: Ken Kansky
#
import numpy as np
from numpy import cos, sin, pi, log, exp, ceil, sqrt, abs
from scipy.signal import convolve2d
from scipy.misc.pilutil import toimage
import PIL.Image as Image


class GaborFilter(object):

    """
      Generates a mask for the Gabor filter with the
      given parameters. Defaults roughly match biological
      filters observed in S1 cells in the visual cortex.

      constrain : If True, the mask will have same dimension
                  as the wavelength
    """

    def __init__(self, wavelength, theta, gamma=2, a=0.5, constrain=False):

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
        """
        Convolves the input image with the stored Gabor mask.
        The output image is the same dimension as the input, and values
        beyond edges are assumed to have the same pixel value as the
        nearest edge pixel.
        """
        return abs(convolve2d(image, self.mask, mode='same'))

    def show(self):
        """ Plots the wavelet. """
        toimage(self.mask).show()

if __name__ == '__main__':
    gf = GaborFilter(10, 10)

    pic = Image.open("./image.png")
    pix = np.array(pic.getdata().reshape(pic.size[0], pic.size[1], 3))

    gf.convolve('./image.png')
    gf.show()
