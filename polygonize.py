from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from skimage import filters, img_as_float32, io, draw, img_as_ubyte
from skimage.color import rgb2gray
from sklearn.cluster import MiniBatchKMeans
import os
import re


def save_image(path, im):
    return io.imsave(path, img_as_ubyte(im.copy()))

def cluster_img(img: np.ndarray, clusters: int) -> np.ndarray:
    """Form clusters from an input image"""
    # Initialize kmeans model
    kmeans = MiniBatchKMeans(
        n_clusters=clusters,
        max_iter=1,
        batch_size=2560,
        tol=0.0,
        max_no_improvement=5
    )

    # Weights
    COLOR_WEIGHT = 1.5
    INTENSITY_WEIGHT = 1
    POSITION_WEIGHT = 2

    # Apply Gaussian blur to image
    img = filters.gaussian(img, sigma=3, multichannel=True)

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

    return res_image

def BowyerWatson (oldTriangulation, points, newpoints):
    triangulation = oldTriangulation.copy()
    superpoints = np.array([[0,0],[1000,0],[0,1000]])
    points = np.concat(points,newpoints,superpoints)
    triangulation.append([-1,-2,-3])
    for i in range(points.shape[0]):
        point = points[i]
        badTriangles = []
        for triangle in triangulation:
            pass
    return triangulation


def triangulate(original_img: np.ndarray, clustered_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Performs Delaunay triangulation on a segmented (clustered) image."""
    # Edge detection on res
    res = filters.sobel(rgb2gray(clustered_img))
    res[res > 0.1] = 1
    res[res <= 0.1] = 0

    # Select a random subset of edge points
    j, i = np.nonzero(res)
    PERCENT_TO_SELECT = 0.01
    selected_indices = np.random.choice(len(i), int(
        np.floor(PERCENT_TO_SELECT * len(i))), replace=False)

    i = i[selected_indices]
    j = j[selected_indices]

    # Add image border points
    print(j)


    height = original_img.shape[0]  # j
    width = original_img.shape[1]  # i
    
    numDiv = 40
    top_border_points_i = []
    bottom_border_points_i = []
    left_border_points_j = []
    right_border_points_j = []
    for k in range(numDiv+1):
        top_border_points_i.append((int)((k/numDiv)*(width-1)))
        bottom_border_points_i.append((int)((k/numDiv)*(width-1)))
        left_border_points_j.append((int)((k/numDiv)*(height-1)))
        right_border_points_j.append((int)((k/numDiv)*(height-1)))
    top_border_points_i = np.array(top_border_points_i)
    bottom_border_points_i = np.array(bottom_border_points_i)
    left_border_points_j = np.array(left_border_points_j)
    right_border_points_j = np.array(right_border_points_j)
    top_border_points_j = np.zeros(numDiv+1)
    bottom_border_points_j = np.full(numDiv+1, height-1)
    left_border_points_i = np.zeros(numDiv+1)
    right_border_points_i = np.full(numDiv+1, width-1)

    j = np.concatenate(
        [
            j,
            top_border_points_j,
            bottom_border_points_j,
            left_border_points_j,
            right_border_points_j
        ],
        axis=0
    )
    print(j.shape)

    i = np.concatenate(
        [
            i,
            top_border_points_i,
            bottom_border_points_i,
            left_border_points_i,
            right_border_points_i
        ],
        axis=0
    )
    print(i.shape)

    points = np.concatenate([i[:, None], j[:, None]], axis=1)

    # Delaunay triangulation over edge points
    tri = Delaunay(points)

    plt.triplot(points[:, 0], points[:, 1], tri.simplices)
    plt.plot(points[:, 0], points[:, 1], 'o')

    plt.imshow(original_img)
    plt.show()

    return points, tri.simplices


def visualize(img: np.ndarray, points: np.ndarray, simplices: np.ndarray) -> None:
    """Perform shading of triangulation and visualize results."""
    colors = []
    for i in range(len(simplices)):
        triangle = points[simplices[i]]
        pixels = img[draw.polygon(triangle[:,1], triangle[:,0])]
        color = np.average(pixels, 0)
        img[draw.polygon(triangle[:,1], triangle[:,0])] = color
    plt.imshow(img)
    plt.show()
    return img

def polygonize(path: str, clusters: int) -> None:
    """Polygonize a specified image with a specified number of clusters for segmentation."""
    # Clustering on original image
    img = img_as_float32(io.imread(path))
    res = cluster_img(img, clusters)

    # Delaunay triangulation
    points, simplices = triangulate(img, res)

    # Visualize
    result = visualize(img, points, simplices)
    save_image("results" + os.sep + re.split(('/|\\\\'),path)[-1], result)
