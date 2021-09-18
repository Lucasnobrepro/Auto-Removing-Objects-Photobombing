import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


from libs.pconv_model import PConvUnet

#Both msk and img should be of order (512,512,3) exactly or else use Image Chunker 
img = 'image.jpeg'  #Path of image and mask
msk= 'mask.jpg'

im= cv2.imread(img)eqwewqe
mk= cv2.imread(msk)

im = cv2.resize(im, (512, 512), interpolation = cv2.INTER_AREA)
mk = cv2.resize( mk, (512, 512), interpolation = cv2.INTER_AREA)

mk= np.array(mk)/255
im= np.array(im)/255

mk= mk.reshape(-1,512,512,3)
im= im.reshape(-1,512,512,3)  #The model takes 4D input

model = PConvUnet(vgg_weights=None, inference_only=True)
model.load(r"pconv_imagenet.26-1.07.h5", train_bn=False) #See more about weight in readme
pred_imgs = model.predict([im,mk])

def plot_images(images, s=5):
    _, axes = plt.subplots(1, len(images), figsize=(s*len(images), s))
    if len(images) == 1:
        axes = [axes]
    for img, ax in zip(images, axes):
        ax.imshow(img)
    plt.show()
plot_images(pred_imgs)
import cv2
cv2.imwrite('inpainted.jpg', pred_imgs)