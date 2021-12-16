import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_float32, io
from sklearn.cluster import MiniBatchKMeans
from skimage import filters


def plot_clusters(img: np.ndarray) -> np.ndarray:
    """Form clusters from an input image"""
    # Initialize kmeans model
    kmeans = MiniBatchKMeans(
        n_clusters=10,
        max_iter=50,
        batch_size=1024,  # try 2560
        tol=0.0,
        max_no_improvement=5
    )

    # Weights
    COLOR_WEIGHT = 1
    INTENSITY_WEIGHT = 1 / 1.5
    POSITION_WEIGHT = 2.5

    # Apply Gaussian blur to image
    img = filters.gaussian(img, sigma=2, multichannel=True)

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


def polygonize(path: str) -> None:
    clustered_image = plot_clusters(img_as_float32(io.imread(path)))
     
