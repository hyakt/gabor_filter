import numpy as np
from numpy import cos, sin, pi, log, exp, ceil, sqrt, abs, radians
from scipy.signal import convolve2d
from scipy.misc.pilutil import toimage
import PIL.Image as Image


class GaborFilter(object):

    def __init__(self, wavelength, theta, gamma=2, a=0.5):

        # 最長軸のためのピクセル寸法を計算する
        eta = -0.5 / (a * wavelength) ** 2
        size = 1 + 2 * int(ceil(sqrt(log(0.05) * max(1, gamma ** 2) / eta)))

        self.mask = np.zeros((size, size), 'float32')

        # フィルターの生成
        for x in range(size):
            for y in range(size):
                tx = x - (size - 1) / 2.0
                ty = y - (size - 1) / 2.0
                rx = tx * cos(theta) + ty * sin(theta)
                ry = -1 * tx * sin(theta) + ty * cos(theta)
                ex = exp(eta * (rx ** 2 + ry ** 2 / gamma ** 2))
                self.mask[x, y] = ex * sin(2 * pi * rx / wavelength)

        # 平均0と大きさ1のフィルターを正規化
        self.mask -= self.mask.sum() / self.mask.size
        self.mask /= sqrt((self.mask * self.mask).sum())

    def convolve(self, image):
        print("convert image")
        return abs(convolve2d(image, self.mask, mode='same'))

    def show(self):
        print("wave show call")
        toimage(self.mask).save("./wave.png")

if __name__ == '__main__':
    gf = GaborFilter(3, radians(0))

    img = Image.open("./img.png")
    imgarr = np.asarray(img)

    convarr = gf.convolve(imgarr)
    convimg = Image.fromarray(convarr)
    convimg.convert('RGB').save('./convimg.png')

    gf.show()
    print("end")
