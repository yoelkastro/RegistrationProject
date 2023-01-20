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

tf_param, _, _ = cpd.registration_cpd(H, M, "affine")

fixed_image_name = 'fixed_image.tif'
moving_image_name = 'moving_image.tif'
dataset_path = '/home/yoelkastro/Desktop/2022-2023/3rd Year Project/'

fixed_img_file_path = os.path.join(dataset_path, fixed_image_name)
moving_img_file_path = os.path.join(dataset_path, moving_image_name)

fixed_wsi_reader = WSIReader.open(input_img=fixed_img_file_path)
moving_wsi_reader = WSIReader.open(input_img=moving_img_file_path)

thumb_level = 8
fixed_image = fixed_wsi_reader.slide_thumbnail(resolution=thumb_level, units="level")
moving_image = moving_wsi_reader.slide_thumbnail(resolution=thumb_level, units="level")
fixed_mask = fixed_wsi_reader.tissue_mask(resolution=thumb_level, units="level").img
moving_mask = moving_wsi_reader.tissue_mask(resolution=thumb_level, units="level").img

# extract tissue region at level 6
x_fixed, y_fixed, w_fixed, h_fixed = cv2.boundingRect(fixed_mask)
x_moving, y_moving, w_moving, h_moving = cv2.boundingRect(moving_mask)
translation_transform_level8 = np.array(
    [
        [1, 0, (x_fixed - x_moving)],
        [0, 1, (y_fixed - y_moving)],
        [0, 0, 1],
    ], dtype=float
)

thumb_level = 6      # level 6
scale_factor = 2**8     # this factor is used to upscale to level 0 from level 8
orig_fixed_roi = fixed_wsi_reader.read_region((x_fixed*scale_factor, y_fixed*scale_factor), thumb_level, (w_fixed*4, h_fixed*4))
orig_moving_roi = moving_wsi_reader.read_region((x_moving*scale_factor, y_moving*scale_factor), thumb_level, (w_moving*4, h_moving*4))

# Preprocessing fixed and moving images
fixed_roi = preprocess_image(orig_fixed_roi)
moving_roi = preprocess_image(orig_moving_roi)
fixed_roi, moving_roi = match_histograms(fixed_roi, moving_roi)

before_reg_moving = cv2.warpAffine(
    moving_roi, np.eye(2, 3), orig_fixed_roi.shape[:2][::-1]
)


newRot = np.ndarray(shape=(2, 3), dtype=float)
newRot[0] = np.append(tf_param.b[0], [tf_param.t[0]], axis=0)#tf_param.t[0]
newRot[1] = np.append(tf_param.b[1], [tf_param.t[1]], axis=0)

dfbr_registered_image = cv2.warpAffine(
    orig_moving_roi, newRot, orig_fixed_roi.shape[:2][::-1]
)
dfbr_registered_mask = cv2.warpAffine(
    moving_mask, newRot, orig_fixed_roi.shape[:2][::-1]
)

before_overlay = np.dstack((before_reg_moving, fixed_roi, before_reg_moving))
dfbr_overlay = np.dstack((dfbr_registered_image[:,:,0], fixed_roi, dfbr_registered_image[:,:,0]))

_, axs = plt.subplots(1, 2, figsize=(15, 10))
axs[0].imshow(before_overlay, cmap="gray")
axs[0].set_title("Overlay Before Registration")
axs[1].imshow(dfbr_overlay, cmap="gray")
axs[1].set_title("Overlay After Transform")
plt.show()

forward_translation = np.array(
    [
        [1, 0, -x_fixed],
        [0, 1, -y_fixed],
        [0, 0, 1],
    ]
)
inverse_translation = np.array(
    [
        [1, 0, x_fixed],
        [0, 1, y_fixed],
        [0, 0, 1],
    ]
)
dfbr_transform_level8 = newRot[0:-1] * np.array([[1, 1, 1/4], [1, 1, 1/4], [1, 1, 1]])
image_transform = inverse_translation @ dfbr_transform_level8 @ forward_translation
final_reg_transform = image_transform @ translation_transform_level8

registered_image = cv2.warpAffine(
    moving_image, final_reg_transform[0:-1], fixed_image.shape[:2][::-1]
)

_, axs = plt.subplots(1, 2, figsize=(15, 10))
axs[0].imshow(fixed_image, cmap="gray")
axs[0].set_title("Fixed Image")
axs[1].imshow(registered_image, cmap="gray")
axs[1].set_title("Registered Image")
plt.show()