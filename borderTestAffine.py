import cv2
import pandas
from random import randint
from random import sample
import numpy as np
from probreg import cpd
from probreg import callbacks
import matplotlib.pyplot as plt


dataset = pandas.read_csv("dataset/dataset_medium.csv")
dataset = dataset[dataset["status"] == "training"]			# Get a list of all training images in the dataset

ind = randint(0, len(dataset))							    # Index of images to register on list

# Significant indices: 124, 90
print(ind)

src_img = cv2.imread("dataset/images/" + dataset["src img"].loc[dataset.index[ind]], cv2.IMREAD_COLOR)
trgt_img = cv2.imread("dataset/images/" + dataset["target img"].loc[dataset.index[ind]], cv2.IMREAD_COLOR)

src_edge = cv2.Canny(src_img, 30, 200)
trgt_edge = cv2.Canny(trgt_img, 30, 200)

src_cont, src_hierarchy = cv2.findContours(src_edge, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
trgt_cont, trgt_hierarchy = cv2.findContours(trgt_edge, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


src_cont_np = [c[0] for c in src_cont]
trgt_cont_np = [c[0] for c in trgt_cont]


numCont = 1000

src_cont = np.array(sample([c[0][0] for c in src_cont], numCont))
trgt_cont = np.array(sample([c[0][0] for c in trgt_cont], numCont))


'''
i = 1
while len(contours[i]) > len(contours[0]) / 2: 
	print(len(contours[i]))
	np.concatenate((allPoints, contours[i]))
	
	i += 1
'''
cv2.drawContours(src_img, src_cont_np, numCont - 1, (0, 255, 0), 3)
cv2.drawContours(trgt_img, trgt_cont_np, numCont - 1, (0, 255, 0), 3)
cv2.imwrite('contoured_target.jpg', trgt_img)
cv2.imwrite('contoured_source.jpg', src_img)


print(src_cont.shape)

print(trgt_cont.shape)
cbs = [callbacks.Plot2DCallback(src_cont, trgt_cont)]
tf_param, _, _ = cpd.registration_cpd(src_cont, trgt_cont, "affine", callbacks=cbs)


newRot = np.ndarray(shape=(2, 3), dtype=float)

newRot[0] = np.append(tf_param.b[0], [tf_param.t[0]], axis=0)
newRot[1] = np.append(tf_param.b[1], [tf_param.t[1]], axis=0)

height, width = src_img.shape[:2]
rotated_image = cv2.warpAffine(src=src_img, M=newRot, dsize=(width, height))							# Apply rotational and translational transform to source image
#rotated_image = cv2.resize(rotated_image, (int(width * tf_param.scale), int(height * tf_param.scale)))	# Apply scaling to source image

cv2.imwrite('rotated_image.jpg', rotated_image)

print(newRot)

plt.show()