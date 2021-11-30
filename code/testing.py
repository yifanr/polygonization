import numpy as np
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale, resize
import os

resultsDir = '..' + os.sep + 'results'
dataDir = '..' + os.sep + 'data'

def load_image(path):
    return img_as_float32(io.imread(path))

def save_image(path, im):
    return io.imsave(path, img_as_ubyte(im.copy()))

test_image = load_image("../data/boats.jpg")
blur_filter = np.ones((3, 3), dtype=np.float32)
# making the filter sum to 1
blur_filter /= np.sum(blur_filter, dtype=np.float32)
blur_image = my_imfilter(test_image, blur_filter)
plt.imshow(blur_image,cmap='gray')
plt.show()
done = save_image(resultsDir + os.sep + 'blur_image.jpg', blur_image)