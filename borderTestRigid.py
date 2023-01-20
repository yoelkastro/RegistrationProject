import cv2
import pandas
from random import randint
from random import sample
import numpy as np
from probreg import cpd
from probreg import callbacks
import matplotlib.pyplot as plt


dataset = pandas.read_csv("dataset/dataset_medium.csv")
#dataset = dataset[dataset["status"] == "training"]			# Get a list of all training images in the dataset

ind =  195#randint(0, len(dataset))							    # Index of images to register on list

# Significant indices:
# 		Large rotation: 			81, 118, 135, 195, 262, 269
#		Significant contour noise: 	358

'''
pandas.set_option('display.max_rows', None)
i = 0
for idx, row in dataset.iterrows():
	print(i, row['src img'], row['target img'])
	i += 1
'''
print(ind)

print(dataset["src img"].loc[dataset.index[ind]])
print(dataset["target img"].loc[dataset.index[ind]])

src_img = cv2.imread("dataset/images/" + dataset["src img"].loc[dataset.index[ind]], cv2.IMREAD_COLOR)
trgt_img = cv2.imread("dataset/images/" + dataset["target img"].loc[dataset.index[ind]], cv2.IMREAD_COLOR) # Get the source and target tissue images

src_edge = cv2.Canny(src_img, 30, 200)
trgt_edge = cv2.Canny(trgt_img, 30, 200)

src_cont, src_hierarchy = cv2.findContours(src_edge, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
trgt_cont, trgt_hierarchy = cv2.findContours(trgt_edge, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)				# Find the contours of the images


src_cont_np = [c[0] for c in src_cont]
trgt_cont_np = [c[0] for c in trgt_cont]


numCont = 10000

src_cont = np.array(sample([c[0][0] for c in src_cont], numCont))
trgt_cont = np.array(sample([c[0][0] for c in trgt_cont], numCont)) 	# Get a random sample of size numCont from the point set representation of the image's contours

cv2.drawContours(src_img, src_cont_np, numCont - 1, (0, 255, 0), 3)
cv2.drawContours(trgt_img, trgt_cont_np, numCont - 1, (0, 255, 0), 3)
cv2.imwrite('contoured_target.jpg', trgt_img)
cv2.imwrite('contoured_source.jpg', src_img)


cbs = [callbacks.Plot2DCallback(src_cont, trgt_cont)]
#tf_param, _, _ = cpd.registration_cpd(src_cont, trgt_cont, "rigid", callbacks=cbs)	# Apply the CPD algorithm to the obtained point sets

pandas.DataFrame(src_cont).to_csv("source.csv")
pandas.DataFrame(trgt_cont).to_csv("target.csv")

'''
newRot = np.ndarray(shape=(2, 3), dtype=float)

newRot[0] = np.append(tf_param.rot[0], [tf_param.t[0]], axis=0)
newRot[1] = np.append(tf_param.rot[1], [tf_param.t[1]], axis=0)

height, width = src_img.shape[:2]
rotated_image = cv2.warpAffine(src=src_img, M=newRot, dsize=(width, height))							# Apply rotational and translational transform to source image
rotated_image = cv2.resize(rotated_image, (int(width * tf_param.scale), int(height * tf_param.scale)))	# Apply scaling to source image

cv2.imwrite('rotated_image.jpg', rotated_image)

print(newRot)

plt.show()
'''