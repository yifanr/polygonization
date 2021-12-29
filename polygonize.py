import os
import re
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.npyio import save
from scipy.spatial import Delaunay
from skimage import draw, filters, img_as_float32, img_as_ubyte, io
from skimage.color import rgb2gray, rgba2rgb
from sklearn.cluster import MiniBatchKMeans
from skimage.feature import greycomatrix, greycoprops

def save_image(path, im):
    """Save an image."""
    return io.imsave(path, img_as_ubyte(im.copy()))


def cluster_img(img: np.ndarray, clusters: int) -> np.ndarray:
    """Form clusters from an input image"""
    # Initialize kmeans model
    kmeans = MiniBatchKMeans(
        n_clusters=clusters,
        max_iter=50,
        batch_size=2560,
        tol=0.0,
        max_no_improvement=5
    )
    gray_img = np.mean(img, axis=2)
    binned_img = (7.999 * gray_img).astype(int)

    # Weights
    COLOR_WEIGHT = 1.5
    INTENSITY_WEIGHT = 1.2
    POSITION_WEIGHT = 0.8
    CONTRAST_WEIGHT = 0.2
    CORRELATION_WEIGHT = 0.3
    ASM_WEIGHT = 0.5

    s = (img.size**0.5)/2000
    halfWindow = max((int)(5*s), 2)
    contrast = np.zeros((img.shape[0]-2*halfWindow, img.shape[1]-2*halfWindow))
    correlation = np.zeros((img.shape[0]-2*halfWindow, img.shape[1]-2*halfWindow))
    asm = np.zeros((img.shape[0]-2*halfWindow, img.shape[1]-2*halfWindow))

    print("Calculating textures")

    # calculate glcm textures for each pixel
    for i in range(img.shape[0] - 2*halfWindow):
        for j in range(img.shape[1] - 2*halfWindow):
            patch = binned_img[i:i + 2*halfWindow + 1,j:j + 2*halfWindow + 1]
            glcm = greycomatrix(patch, [1], [0], levels=8)
            contrast[i,j] = greycoprops(glcm, 'contrast')
            correlation[i,j] = greycoprops(glcm, 'correlation')
            asm[i,j] = greycoprops(glcm, 'ASM')
    # pad edges
    contrast = np.pad(contrast, halfWindow, mode="edge")
    correlation = np.pad(correlation, halfWindow, mode="edge")
    asm = np.pad(asm, halfWindow, mode="edge")
    plt.imshow(contrast/np.mean(contrast))
    plt.show()
    plt.imshow(correlation/np.mean(correlation))
    plt.show()
    plt.imshow(asm/np.mean(asm))
    plt.show()
    #normalize and expand dims
    contrast = np.expand_dims(contrast/np.mean(contrast), 2)
    correlation = np.expand_dims(correlation/np.mean(correlation), 2)
    asm = np.expand_dims(asm/np.mean(asm), 2)


    # Apply Gaussian blur to image
    img = filters.gaussian(img, sigma=s, multichannel=True)

    # Calculate features
    color_mean = np.mean(img)
    img /= color_mean

    coordinates = np.indices(img.shape[0:2]).transpose((1, 2, 0)).astype(float)
    coordinates /= np.mean(coordinates)
    intensity = np.expand_dims(gray_img, 2) / color_mean
    X = np.concatenate((img * COLOR_WEIGHT, intensity * INTENSITY_WEIGHT, contrast * CONTRAST_WEIGHT, correlation * CORRELATION_WEIGHT, asm * ASM_WEIGHT, coordinates * POSITION_WEIGHT,), axis=2)
    X = X.reshape(-1, 9)

    print("Beginning to fit model")

    kmeans.fit(X)

    print("Model fit")
    clustered = np.reshape(kmeans.cluster_centers_[kmeans.labels_][:,0:7], (img.shape[0],img.shape[1],7))
    res_image = clustered[:, :, 0:3]
    res_image = res_image * color_mean / COLOR_WEIGHT

    # # Plot to subplots
    # f, axes = plt.subplots(1, 2)
    # axes[0].imshow(img * color_mean)
    # axes[1].imshow(res_image)

    # plt.show()
    save_image("results"       + os.sep + "kmeans.jpg", res_image)

    return clustered


def triangulate(original_img: np.ndarray, clustered_img: np.ndarray, vertices: int, percent: float) -> Tuple[np.ndarray, np.ndarray]:
    """Performs Delaunay triangulation on a segmented (clustered) image."""
    s = (clustered_img.size**0.5)/4000
    print(clustered_img.shape)
    # Apply Gaussian blur to image
    clustered_img = filters.gaussian(clustered_img, sigma=s, multichannel=True)
    # Edge detection on res
    res = filters.sobel(clustered_img[:,:,0])
    res += filters.sobel(clustered_img[:,:,1])
    res += filters.sobel(clustered_img[:,:,2])
    res += filters.sobel(clustered_img[:,:,3])
    res += filters.sobel(clustered_img[:,:,4])
    res += filters.sobel(clustered_img[:,:,5])
    res += filters.sobel(clustered_img[:,:,6])
    print(np.max(res))
    #percentile = 100-(((vertices**.8)/res.size)*50000)
    percentile = 100 - percent
    print(percentile)
    cutoff = np.percentile(res, percentile)
    print(cutoff)
    # cutoff = 0.1
    print(res[res>cutoff].size)
    res[res <= cutoff] = 0

    # Select a random subset of edge points
    j, i = np.nonzero(res)
    print("Randomly sampling from " + str(len(i)) + " edge points")
    selected_indices = np.random.choice(len(i), min(len(i), (int)(1*vertices)), replace=False)

    i = i[selected_indices]
    j = j[selected_indices]
    # i = np.append(i, np.random.choice(clustered_img.shape[1], (int)(0.05*vertices)))
    # j = np.append(j, np.random.choice(clustered_img.shape[0], (int)(0.05*vertices)))

    # Add image border points
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
    points = np.concatenate([i[:, None], j[:, None]], axis=1)

    # Delaunay triangulation over edge points
    tri = Delaunay(points)

    plt.triplot(points[:, 0], points[:, 1], tri.simplices)
    plt.plot(points[:, 0], points[:, 1], 'o')

    plt.imshow(original_img)

    return points, tri.simplices


def visualize(img: np.ndarray, points: np.ndarray, simplices: np.ndarray, average: bool) -> None:
    """Perform shading of triangulation and visualize results."""
    for i in range(len(simplices)):
        triangle = points[simplices[i]]
        pixels = img[draw.polygon(triangle[:, 1], triangle[:, 0])]
        if (average):
            color = np.average(pixels, 0)
        else:
            centroid = np.average(triangle, axis=0).astype(int)
            color = img[centroid[1], centroid[0]]
        img[draw.polygon(triangle[:, 1], triangle[:, 0])] = color

    return img


def polygonize(path: str, clusters: int, vertices: int, percent: float, average: bool) -> np.ndarray:
    """Polygonize a specified image with a specified number of clusters for segmentation."""
    # Convert to RGB
    img = io.imread(path)
    channels = img.shape[2]
    if (channels == 1):
        print("Error: only color images are supported")
        return
    elif (channels == 3):
        print("Detected RGB image")
    elif (channels == 4):
        print("Detected RGBA image, converting to RGB")
        img = rgba2rgb(img)
    else:
        print("Error: unknown image format")
        return

    # Clustering on original image
    print("Clustering")
    img = img_as_float32(img)
    res = cluster_img(img, clusters)
    # Delaunay triangulation
    points, simplices = triangulate(img, res, vertices, percent)

    # Visualize
    result = visualize(img, points, simplices, average)
    save_image("results" + os.sep + re.split(('/|\\\\'), path)[-1], result)

    return result
