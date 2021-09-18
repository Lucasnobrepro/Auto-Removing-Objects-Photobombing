import cv2
import numpy as np
import argparse
from config.configClasses import COCO_CLASSES, colors

class yolact():
    def __init__(self, confThreshold=0.3, nmsThreshold=0.3, keep_top_k=200, path_weights='./weights/yolact_base_54_800000.onnx'):
        self.target_size = 550
        self.MEANS = np.array([103.94, 116.78, 123.68], dtype=np.float32).reshape(1, 1, 3)
        self.STD = np.array([57.38, 57.12, 58.40], dtype=np.float32).reshape(1, 1, 3)
        self.net = cv2.dnn.readNet(path_weights)
        self.confidence_threshold = confThreshold
        self.nms_threshold = nmsThreshold
        self.keep_top_k = keep_top_k
        self.conv_ws = [69, 35, 18, 9, 5]
        self.conv_hs = [69, 35, 18, 9, 5]
        self.aspect_ratios = [1, 0.5, 2]
        self.scales = [24, 48, 96, 192, 384]
        self.variances = [0.1, 0.2]
        self.last_img_size = None
        self.priors = self.make_priors()

    def make_priors(self):
        """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """
        if self.last_img_size != (self.target_size, self.target_size):
            prior_data = []

            for conv_w, conv_h, scale in zip(self.conv_ws, self.conv_hs, self.scales):
                for i in range(conv_h):
                    for j in range(conv_w):
                        # +0.5 because priors are in center-size notation
                        cx = (j + 0.5) / conv_w
                        cy = (i + 0.5) / conv_h

                        for ar in self.aspect_ratios:
                            ar = np.sqrt(ar)

                            w = scale * ar / self.target_size
                            h = scale / ar / self.target_size

                            # This is for backward compatability with a bug where I made everything square by accident
                            h = w

                            prior_data += [cx, cy, w, h]

            self.priors = np.array(prior_data).reshape(-1, 4)
 # plt.subplot(222), plt.imshow(objs[:,:,::-1]);# plt.subplot(222), plt.imshow(objs[:,:,::-1]);           self.last_img_size = (self.target_size, self.target_size)
        return self.priors

    def decode(self, loc, priors, img_w, img_h):
        boxes = np.concatenate(
            (
                priors[:, :2] + loc[:, :2] * self.variances[0] * priors[:, 2:],
                priors[:, 2:] * np.exp(loc[:, 2:] * self.variances[1]),
            ),
            1,
        )
        boxes[:, :2] -= boxes[:, 2:] / 2
        # boxes[:, 2:] += boxes[:, :2]

        # crop
        np.where(boxes[:, 0] < 0, 0, boxes[:, 0])
        np.where(boxes[:, 1] < 0, 0, boxes[:, 1])
        np.where(boxes[:, 2] > 1, 1, boxes[:, 2])
        np.where(boxes[:, 3] > 1, 1, boxes[:, 3])

        # decode to img size
        boxes[:, 0] *= img_w
        boxes[:, 1] *= img_h
        boxes[:, 2] = boxes[:, 2] * img_w + 1
        boxes[:, 3] = boxes[:, 3] * img_h + 1
        return boxes

    def detect(self, srcimg, label, limit_obj=0):
        img_h, img_w = srcimg.shape[:2]
        img = cv2.resize(srcimg, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        img = (img - self.MEANS) / self.STD

        blob = cv2.dnn.blobFromImage(img, swapRB=True)
        # Sets the input to the network
        self.net.setInput(blob)
        # Runs the forward pass to get output of the output layers
        loc_data, conf_preds, mask_data, proto_data = self.net.forward(self.net.getUnconnectedOutLayersNames())

        cur_scores = conf_preds[:, 1:]
        num_class = cur_scores.shape[1]
        classid = np.argmax(cur_scores, axis=1)
        # conf_scores = np.max(cur_scores, axis=1)
        conf_scores = cur_scores[range(cur_scores.shape[0]), classid]

        # filte by confidence_threshold
        keep = conf_scores > self.confidence_threshold
        conf_scores = conf_scores[keep]
        classid = classid[keep]
        loc_data = loc_data[keep, :]
        prior_data = self.priors[keep, :]
        masks = mask_data[keep, :]
        boxes = self.decode(loc_data, prior_data, img_w, img_h)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), conf_scores.tolist(), self.confidence_threshold, self.nms_threshold , top_k=self.keep_top_k)

        final_mask = srcimg.copy()
        final_mask[:,:,:] = 255
        objects = srcimg.copy()
        count_obj = 1

        for i in indices:
            idx = i[0]
            left, top, width, height = boxes[idx, :].astype(np.int32).tolist()
                        
            if COCO_CLASSES[classid[idx]+1] == label or label == "all":
                
                cv2.rectangle(objects, (left, top), (left+width, top+height), (0, 0, 255), thickness=1)
                # cv2.putText(image, "Objeto: "+ str(idx) + COCO_CLASSES[classid[idx]+1]+':'+str(round(conf_scores[idx], 2)), 
                #            (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
                cv2.putText(objects, "Objeto: "+ str(count_obj), (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
                # generate mask
                mask = proto_data @ masks[idx, :].reshape(-1,1)
                mask = 1 / (1 + np.exp(-mask))  ###sigmoid

                # Scale masks up to the full image
                mask = cv2.resize(mask.squeeze(), (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                mask = mask > 0.5

                objects[mask] = objects[mask] * 0.5  + np.array(colors[classid[idx]+1]) * 0.5
                # return mask;
                final_mask[mask] = 0
                count_obj+= 1
            
            if limit_obj == 1:
                break
        
        mask_inv = cv2.bitwise_not(final_mask)
               
        return srcimg, objects, final_mask, mask_inv