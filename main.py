import cv2
import numpy as np
import argparse

from yolact_detect import yolact
from libs.pconv_model import PConvUnet
from matplotlib import pyplot as plt

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='YOLACT COCO Evaluation')
    parser.add_argument('--imgpath', default='./data/sheap.jpg', type=str, help='A path to an image to use for display.')
    parser.add_argument('--confThreshold', default=0.5, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    parser.add_argument('--label', default="all", type=str, help='label object to detect')
    args = parser.parse_args()

    # Read Image
    srcimg = cv2.imread(str(args.imgpath))

    # Instace Yolact
    myyolact = yolact()
    image, mask = myyolact.detect(srcimg, str(args.label))

    cv2.imwrite('image.jpg', image)
    cv2.imwrite('mask.jpg', mask)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    # plt.subplot(111), plt.imshow(mask)
    # plt.subplot(122), plt.imshow(image)

    image = cv2.resize(image, (512, 512), interpolation = cv2.INTER_AREA)
    mask = cv2.resize(mask, (512, 512), interpolation = cv2.INTER_AREA)

    mask = np.array(mask)/255
    image = np.array(image)/255

    mask = mask.reshape(-1,512,512,3)
    image = image.reshape(-1,512,512,3)  #The model takes 4D input

    model = PConvUnet(vgg_weights=None, inference_only=True)

    model.load(r"./weights/pconv_imagenet.26-1.07.h5", train_bn=False) #See more about weight in readme
    pred_imgs = model.predict([image,mask])

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

    
    # for i in range(n):
    #     # image = model.predict([image,mask])[0]
    #     print(i)
    #     image = image.reshape(-1,512,512,3)
    #     image = model.predict([image,mask])[0]

    #     cv2.imwrite("./results/out" +str(i)+".jpg", image * 255)
    # cv2.imshow("resultado depois de " + str(n) + "repetições",image)
    # while(1):
    #     k = cv2.waitKey(0) & 0xff
    #     if k == 27:
    #         break
    # cv2.destroyAllWindows()
    # plt.show()

    