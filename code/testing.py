import numpy as np
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale, resize
from skimage import filters, feature, img_as_int
import os
from sklearn.cluster import KMeans
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2

resultsDir = '..' + os.sep + 'results'
dataDir = '..' + os.sep + 'data'

def load_image(path):
    return img_as_float32(io.imread(path))

def save_image(path, im):
    return io.imsave(path, img_as_ubyte(im.copy()))

def get_interest_points(image):
    '''
    Returns interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops
          
    Note: You may decide it is unnecessary to use feature_width in get_interest_points, or you may also decide to 
    use this parameter to exclude the points near image edges.

    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width: the width and height of each local feature in pixels

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    '''

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # These are placeholders - replace with the coordinates of your interest points!
    w = 6
    feature_width = np.sqrt(image.size/10000).astype(int).item()*w
    xs = np.zeros(1)
    ys = np.zeros(1)
    a = 0.04
    s = np.sqrt(image.size/90000)
    print(s)
    blurred = filters.gaussian(image, sigma=s/3.5)
    grad_x = filters.sobel_v(blurred)
    grad_y = filters.sobel_h(blurred)
    x2 = filters.gaussian(grad_x*grad_x, sigma=s*0.8)
    y2 = filters.gaussian(grad_y*grad_y, sigma=s*0.8)
    xy = filters.gaussian(grad_x*grad_y, sigma=s*0.8)
    cornerness = x2*y2-xy*xy - a*(x2+y2)*(x2+y2)
    corners = feature.peak_local_max(cornerness, min_distance=np.floor(s*1).astype(int),num_peaks=2000, threshold_rel=0, exclude_border=feature_width//2+1, indices=True)
    ys = corners[:,0]
    xs = corners[:,1]





    # STEP 1: Calculate the gradient (partial derivatives on two directions).
    # STEP 2: Apply Gaussian filter with appropriate sigma.
    # STEP 3: Calculate Harris cornerness score for all pixels.
    # STEP 4: Peak local max to eliminate clusters. (Try different parameters.)
    
    # BONUS: There are some ways to improve:
    # 1. Making interest point detection multi-scaled.
    # 2. Use adaptive non-maximum suppression.

    return xs, ys
colorWeight = 0.8
intensityWeight = 1
positionWeight = 1
maxHeight = 720
numCentroids = 28
path = 'city.jpg'

test_image = load_image(dataDir + os.sep + path)
scale = max(test_image.shape[0]//maxHeight,1)
newShape = (test_image.shape[1]//scale, test_image.shape[0]//scale)
test_image = cv2.resize(test_image, newShape)*0.999
# s = test_image.shape[0]//200
# kernel = np.ones((s,s),np.float32)/25
# print(np.min(test_image))
# test_image = cv2.filter2D(test_image,-1,kernel)
# save_image(resultsDir + os.sep + 'blurred_image.jpg', test_image)
colorMean = np.mean(test_image)
test_image /= colorMean
plt.imshow(test_image)
plt.show()
shape = test_image.shape
coordinates = np.indices(shape[0:2]).transpose((1,2,0)).astype(float)
coordinates /= np.mean(coordinates)
intensity = np.expand_dims(np.mean(test_image, axis=2),2)/colorMean
input = np.concatenate((test_image*colorWeight,intensity*intensityWeight,coordinates*positionWeight),axis=2)
input = input.reshape(-1,6)
print(input.shape)
kmeans = KMeans(n_clusters=numCentroids).fit(input)
kmeans_image = kmeans.cluster_centers_[kmeans.labels_][:,0:3].reshape(shape)*colorMean
io.imshow(kmeans_image)
io.show()
done = save_image(resultsDir + os.sep + 'kmeans_'+path, kmeans_image)
(x1, y1) = get_interest_points(kmeans_image)
plt.imshow(kmeans_image)
plt.scatter(x1, y1, alpha=0.9, s=3)