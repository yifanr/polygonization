from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from skimage import filters, img_as_float32, io
from skimage.color import rgb2gray
from skimage.feature.orb import ORB
from skimage.feature.peak import peak_local_max
from sklearn.cluster import MiniBatchKMeans


def cluster_img(img: np.ndarray, clusters: int) -> Tuple[np.ndarray, np.ndarray]:
    """Form clusters from an input image"""
    # Initialize kmeans model
    kmeans = MiniBatchKMeans(
        n_clusters=clusters,
        max_iter=50,
        batch_size=1024,  # try 2560
        tol=0.0,
        max_no_improvement=5
    )

    # Weights
    COLOR_WEIGHT = 1.5
    INTENSITY_WEIGHT = 1
    POSITION_WEIGHT = 3

    # Apply Gaussian blur to image
    img = filters.gaussian(img, sigma=7, multichannel=True)

    # Calculate features
    color_mean = np.mean(img)
    img /= color_mean

    coordinates = np.indices(img.shape[0:2]).transpose((1, 2, 0)).astype(float)
    coordinates /= np.mean(coordinates)
    intensity = np.expand_dims(np.mean(img, axis=2), 2) / color_mean
    X = np.concatenate((img * COLOR_WEIGHT, intensity *
                        INTENSITY_WEIGHT, coordinates * POSITION_WEIGHT), axis=2)
    X = X.reshape(-1, 6)

    print("Beginning to fit model")

    kmeans.fit(X)

    print("Model fit")

    # res_image = kmeans.cluster_centers_[
    #   kmeans.labels_][:, 0:3].reshape(img.shape) * color_mean
    res_image = kmeans.cluster_centers_[kmeans.labels_][:, 0:3]
    res_image = np.reshape(res_image, img.shape) * color_mean / COLOR_WEIGHT

    # # Plot to subplots
    # f, axes = plt.subplots(1, 2)
    # axes[0].imshow(img * color_mean)
    # axes[1].imshow(res_image)

    # plt.show()

    return res_image, kmeans.labels_

def BowyerWatson (oldTriangulation, points, newpoints):
    triangulation = oldTriangulation.copy()
    superpoints = np.array([[0,0],[1000,0],[0,1000]])
    points = np.concat(points,newpoints,superpoints)
    triangulation.append([-1,-2,-3])
    for i in range(points.shape[0]):
        point = points[i]
        badTriangles = []
        for triangle in triangulation:

    return triangulation

def polygonize(path: str, clusters: int) -> None:
    # Clustering on original image
    img = img_as_float32(io.imread(path))
    res, labels = cluster_img(img, clusters)
    labels = np.reshape(labels, img.shape[0: 2])

    # ORB feature detector
    orb = ORB(
        n_keypoints=50,
        harris_k=0.4
    )

    # Triangulate each cluster individually
    for cluster in range(0, clusters):
        print("Triangulating cluster %d out of %d", cluster, clusters)

        current_cluster = rgb2gray(res)
        current_cluster[labels == cluster] = 1
        current_cluster[labels != cluster] = 0

        current_cluster = filters.sobel(current_cluster)

        current_cluster[current_cluster > 0.1] = 1
        current_cluster[current_cluster <= 0.1] = 0

        plt.imshow(current_cluster, cmap='gray')
        plt.show()

        j, i = np.nonzero(current_cluster)
        PERCENT_TO_SELECT = 0.005
        selected_indices = np.random.choice(len(i), int(
            np.floor(PERCENT_TO_SELECT * len(i))), replace=False)

        i = i[selected_indices]
        j = j[selected_indices]

        points = np.concatenate([i[:, None], j[:, None]], axis=1) 

        # Delaunay
        tri = Delaunay(points)

        plt.triplot(points[:, 0], points[:, 1], tri.simplices)
        plt.plot(points[:, 0], points[:, 1], 'o')

    plt.imshow(res)
    plt.show()
