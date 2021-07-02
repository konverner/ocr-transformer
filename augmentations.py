import numpy as np
import cv2


class Vignetting(object):
    def __init__(self,
                 ratio_min_dist=0.2,
                 range_vignette=(0.2, 0.8),
                 random_sign=False):
        self.ratio_min_dist = ratio_min_dist
        self.range_vignette = np.array(range_vignette)
        self.random_sign = random_sign

    def __call__(self, X, Y=None):
        h, w = X.shape[:2]
        min_dist = np.array([h, w]) / 2 * np.random.random() * self.ratio_min_dist

        # create matrix of distance from the center on the two axis
        x, y = np.meshgrid(np.linspace(-w / 2, w / 2, w), np.linspace(-h / 2, h / 2, h))
        x, y = np.abs(x), np.abs(y)

        # create the vignette mask on the two axis
        x = (x - min_dist[0]) / (np.max(x) - min_dist[0])
        x = np.clip(x, 0, 1)
        y = (y - min_dist[1]) / (np.max(y) - min_dist[1])
        y = np.clip(y, 0, 1)

        # then get a random intensity of the vignette
        vignette = (x + y) / 2 * np.random.uniform(*self.range_vignette)
        vignette = np.tile(vignette[..., None], [1, 1, 3])

        sign = 2 * (np.random.random() < 0.5) * (self.random_sign) - 1
        X = X * (1 + sign * vignette)
        return X


class LensDistortion(object):
    def __init__(self, d_coef=(0.15, 0.05, 0.1, 0.1, 0.05)):
        self.d_coef = np.array(d_coef)

    def __call__(self, X):
        # get the height and the width of the image
        h, w = X.shape[:2]

        # compute its diagonal
        f = (h ** 2 + w ** 2) ** 0.5

        # set the image projective to carrtesian dimension
        K = np.array([[f, 0, w / 2],
                      [0, f, h / 2],
                      [0, 0, 1]])

        d_coef = self.d_coef * np.random.random(5)  # value
        d_coef = d_coef * (2 * (np.random.random(5) < 0.5) - 1)  # sign
        # Generate new camera matrix from parameters
        M, _ = cv2.getOptimalNewCameraMatrix(K, d_coef, (w, h), 0)

        # Generate look-up tables for remapping the camera image
        remap = cv2.initUndistortRectifyMap(K, d_coef, None, M, (w, h), 5)

        # Remap the original image to a new image
        X = cv2.remap(X, *remap, cv2.INTER_LINEAR)
        return X


class UniformNoise(object):
    def __init__(self, low=-50, high=50):
        self.low = low
        self.high = high

    def __call__(self, X):
        noise = np.random.uniform(self.low, self.high, X.shape)
        X = X + noise
        return X


class Cutout(object):
    def __init__(self,
                 min_size_ratio,
                 max_size_ratio,
                 channel_wise=False,
                 crop_target=True,
                 max_crop=1,
                 replacement=0):
        self.min_size_ratio = np.array(list(min_size_ratio))
        self.max_size_ratio = np.array(list(max_size_ratio))
        self.channel_wise = channel_wise
        self.max_crop = max_crop
        self.replacement = 1

    def __call__(self, X):
        size = np.array(X.shape[:2]) * 0.01
        mini = self.min_size_ratio * size
        maxi = self.max_size_ratio * size

        for _ in range(self.max_crop):
            # random size
            if mini[0] == maxi[0]:
                maxi[0] += 1
            h = np.random.randint(mini[0], maxi[0])
            if mini[1] == maxi[1]:
                maxi[1] += 1
            w = np.random.randint(mini[1], maxi[1])
            # random place
            shift_h = np.random.randint(0, abs(size[0] - h) + 1)
            shift_w = np.random.randint(0, abs(size[1] - w) + 1)
            if self.channel_wise:
                c = np.random.randint(0, X.shape[-1])
                X[shift_h:shift_h + h, shift_w:shift_w + w, c] = self.replacement
            else:
                X[shift_h:shift_h + h, shift_w:shift_w + w] = self.replacement

        return X
