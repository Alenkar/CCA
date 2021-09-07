import cv2
import numpy as np
from sklearn.cluster import KMeans
from time import time

np.random.seed(0)


class CCAQI:
    def __init__(self, n_colours: int = 8):
        """
        CCAQI is a Cluster Color Analyse for a Quantization of Image.
        CCAQI uses algorithm k-means to predict centroids of selected number of colors.

        :param n_colours: the number of clusters that are the number of colors to quantization.
        """
        self.n_colours = n_colours
        self.k_means = KMeans(n_clusters=n_colours, random_state=0)

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
        x = np.array(image, dtype=np.float64) / 255
        x = np.reshape(x, (w * h, c))
        t2 = time()
        self.__print_timing(t1, t2)
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
        self.__print_timing(t1, t2)
        return fitted_k_means

    def __get_labels(self, x: np.array):
        """
        Predict the nearest cluster for each sample by x and return their label names.
        :param x: normalised representation of the input image.
        :return: label names as np.array.
        """
        print("Predict labels", end="")
        t1 = time()
        labels = self.k_means.predict(x)
        t2 = time()
        self.__print_timing(t1, t2)
        return labels

    def __create_image(self, labels: np.array, shape: tuple):
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
        for i in range(w):
            for j in range(h):
                image[i][j] = self.k_means.cluster_centers_[labels[label_idx]]
                label_idx += 1
        t2 = time()
        self.__print_timing(t1, t2)
        return image

    def processing(self, image: np.array):
        """
        Processing the input image with a k-means fitting and prediction algorithm
        to extract the cluster centers that are used to generate the compressed image.
        :param image: input image
        :return: quantized image.
        """
        x = self.__prepare_image(image)
        self.k_means = self.__fit(x)
        labels = self.__get_labels(x)
        image = self.__create_image(labels, image.shape[:2])
        return image


img_path = "data/test.png"
img = cv2.imread(img_path)
img = cv2.resize(img, (640, 360))

ccaqi = CCAQI(n_colours=8)

quantizated_img = ccaqi.processing(img)

cv2.imshow("default", img)
cv2.imshow("quantizated_image", quantizated_img)
cv2.waitKey(0)
