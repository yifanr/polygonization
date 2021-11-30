import numpy as np
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale, resize
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

colorWeight = 1
intensityWeight = 1
positionWeight = 1
test_image = load_image(dataDir + os.sep + 'boats.jpg')
scale = max(test_image.shape[0]//1080,1)
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
kmeans = KMeans(n_clusters=16).fit(input)
kmeans_image = kmeans.cluster_centers_[kmeans.labels_][:,0:3].reshape(shape)*colorMean
io.imshow(kmeans_image)
io.show()
done = save_image(resultsDir + os.sep + 'kmeans_image.jpg', kmeans_image)