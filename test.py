import cv2
import pandas
from random import randint
from random import sample
import numpy as np
from probreg import cpd
from probreg import callbacks
import matplotlib.pyplot as plt
import os
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools.registration.wsi_registration import match_histograms, DFBRegister, apply_bspline_transform, estimate_bspline_transform
from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
from scipy import ndimage
from skimage import color, exposure, measure, morphology

def preprocess_image(image):
    """This function converts the RGB image to grayscale image and
    improves the contrast by linearly rescaling the values.
    """
    image = color.rgb2gray(image)
    image = exposure.rescale_intensity(
        image, in_range=tuple(np.percentile(image, (0.5, 99.5)))
    )
    image = image * 255
    return image.astype(np.uint8)

H = np.load('/home/yoelkastro/Desktop/2022-2023/3rd Year Project/fixed_image_land.npy')
M = np.load('/home/yoelkastro/Desktop/2022-2023/3rd Year Project/moving_image_land.npy')

cbs = [callbacks.Plot2DCallback(H, M)]
tf_param, a, b = cpd.registration_cpd(H, M, "affine", callbacks=cbs)
plt.show()

print(a)
print(b)
print(tf_param)


fixed_image_name = 'fixed_image.tif'
moving_image_name = 'moving_image.tif'
dataset_path = '/home/yoelkastro/Desktop/2022-2023/3rd Year Project/reg_visualization_tool/data'


newRot = np.ndarray(shape=(3, 3), dtype=float)

newRot[0] = np.append(tf_param.b[0], [tf_param.t[0]], axis=0)#tf_param.t[0]
newRot[1] = np.append(tf_param.b[1], [tf_param.t[1]], axis=0)
newRot[2] = np.array([0, 0, 1])

n = newRot[0][1]
newRot[0][1] = newRot[1][0]
newRot[1][0] = n

np.save("/home/yoelkastro/Desktop/2022-2023/3rd Year Project/reg_visualization_tool/data/transform.npy",newRot)