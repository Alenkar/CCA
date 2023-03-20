import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from time import time

np.random.seed(0)


class CCAQI:
    def __init__(self, n_colours: int = 8, pixel_size: int = 8, verbose=None):
        """
        CCAQI is a Cluster Color Analyse for a Quantization of Image.
        CCAQI uses algorithm k-means to predict centroids of selected number of colors.

        :param pixel_size:
        :param n_colours: the number of clusters that are the number of colors to quantization.
        """
        self.verbose = verbose
        self.n_colours = n_colours + 1  # add once, because white background
        self.pixel_size = pixel_size
        self.w_size = None
        self.h_size = None
        self.default = None
        self.result = None
        self.labels = None
        self.cluster_centers = None
        self.k_means = KMeans(n_clusters=n_colours, random_state=0)

    def set_config(self):
        self.n_colours = 0
        self.w_size = 0
        self.h_size = 0

    @staticmethod
    def __print_timing(t1: float, t2: float):
        """
        Function for displaying the calculation time to the console.

        :param t1: calculations start time.
        :param t2: calculations end time.
        """
        print(" | Time: {:.3}ms, FPS: {:.2}".format(t2 - t1, 1 / (t2 - t1)))

    def __prepare_image(self, image: np.array):
        """
        Reshaping image to vector view and data normalisation of MinMax.

        :param image: input image represented an np.array matrix.
        :return: vector np.array.
        """
        print("Reshape image and MinMax", end="")
        t1 = time()
        w, h, c = image.shape
        x = np.array(image, dtype=np.float64)# / 255
        x = np.reshape(x, (w * h, c))
        t2 = time()
        if self.verbose:
            self.__print_timing(t1, t2)
        else:
            print()
        return x

    def __fit(self, x: np.array):
        """
        Compute k-means clustering.
        :param x: normalised representation of the input image.
        :return: fitted estimator.
        """
        print("Fitting K-Means", end="")
        t1 = time()
        x = np.random.permutation(x)
        fitted_k_means = self.k_means.fit(x[:2000])
        t2 = time()
        if self.verbose:
            self.__print_timing(t1, t2)
        else:
            print()

        return fitted_k_means.cluster_centers_.astype(np.uint8)

    def __get_labels(self, x: np.array):
        """
        Predict the nearest cluster for each sample by x and return their label names.
        :param x: normalised representation of the input image.
        :return: label names as np.array.
        """
        print("Predict labels", end="")
        t1 = time()
        self.labels = self.k_means.predict(x)
        t2 = time()
        if self.verbose:
            self.__print_timing(t1, t2)
        else:
            print()

    def set_labels(self, labels: np.array):
        self.labels = labels
        self.result = self.__create_image(self.result.shape[:2])

    def __create_image(self, shape: tuple):
        """
        Generation of a quantized image from k-means cluster centers.
        :param labels: label names from k-means.
        :param shape: shape of input image.
        :return:
        """
        print("Create quantized image", end="")
        t1 = time()
        w, h = shape
        image = np.zeros((w, h, 3))
        label_idx = 0

        cluster_centers = self.cluster_centers

        for i in range(w):
            for j in range(h):
                image[i][j] = self.cluster_centers[self.labels[label_idx]]
                label_idx += 1
        t2 = time()

        if self.verbose:
            self.__print_timing(t1, t2)
        else:
            print()
        return image.astype(np.uint8)

    def __pixelization(self, image):
        image = Image.fromarray(image)

        image = image.resize((image.size[0] // self.pixel_size, image.size[1] // self.pixel_size), Image.Resampling.NEAREST)
        image = image.resize((image.size[0] * self.pixel_size, image.size[1] * self.pixel_size), Image.Resampling.NEAREST)
        pixels = np.array(image)
        for i in range(0, image.size[1], self.pixel_size):
            for j in range(0, image.size[0], self.pixel_size):
                for r in range(self.pixel_size):
                    pixels[i + r, j] = (0, 0, 0)
                    pixels[i, j + r] = (0, 0, 0)
        return pixels

    def processing(self, image: np.array):
        """
        Processing the input image with a k-means fitting and prediction algorithm
        to extract the cluster centers that are used to generate the compressed image.
        :param image: input image
        :return: quantized image.
        """
        image = image[:1048, :552]
        self.default = image

        edges = cv2.Canny(image, 30, 150)
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        image = cv2.blur(image, (7, 7))
        # cv2.imshow("result", image)
        # cv2.waitKey(0)

        self.result = self.__pixelization(image)
        x = self.__prepare_image(self.result)
        self.cluster_centers = self.__fit(x)
        self.__get_labels(x)
        self.result = self.__create_image(self.result.shape[:2])

        idx = np.where(edges != 0)
        for i in range(len(idx[0])):
            x = idx[1][i]
            y = idx[0][i]
            self.result = cv2.circle(self.result, (x, y), 4, (0, 0, 0), -1)
        return self.result, self.cluster_centers

    def show_result(self):
        show = np.hstack((self.default, self.result))
        k = 1280 / show.shape[1]
        # show = cv2.resize(show, (0, 0), fx=k,fy=k)
        # cv2.imshow("result", show)
        cv2.imshow("result", self.result)
        cv2.waitKey(0)


# img_path = "data/test.png"
img_path = "/home/neuro/Загрузки/1.jpg"
img = cv2.imread(img_path)
# img = cv2.resize(img, (640, 360))

n_colours = 12
pixel_size = 8


ccaqi = CCAQI(n_colours=n_colours, pixel_size=pixel_size)
# ccaqi.set_config(config)
quantizated_img, labels_idx = ccaqi.processing(img)
print(labels_idx)

new_label = []
for label in labels_idx:
    new_label


ccaqi.show_result()

