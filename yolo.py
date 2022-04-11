# Copyright (c) 2021 Ichiro ITS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from ast import List
from typing import Tuple
import cv2
import numpy as np

class Yolo:
  def __init__(
    self,
    config_path: str,
    weights_path: str,
    names_path: str,
    conf_thresh: float,
    nms_thresh: float,
    use_cuda: bool = False
  ) -> None:
    # save thresholds
    self.ct = conf_thresh
    self.nmst = nms_thresh

    # create net
    self.net = cv2.dnn.readNet(weights_path, config_path)
    print("Loaded weights:", weights_path)
    self.classes = []
    file = open(names_path, 'r')
    for line in file:
      self.classes.append(line.strip())

    # use gpu + CUDA to speed up detections
    if use_cuda:
      self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
      self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    layer_names = self.net.getLayerNames()
    self.output_layers = [layer_names[i-1] for i in self.net.getUnconnectedOutLayers()]

  # runs the network on an image
  def get_inf(self, img: np.ndarray) -> np.ndarray:
    # construct a blob
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # get response
    self.net.setInput(blob)
    layer_outputs = self.net.forward(self.output_layers)
    return layer_outputs

  # filters the layer output by conf, nms and id
  def thresh(self, layer_outputs: np.ndarray, width: float, height: float) -> Tuple:
    # some lists
    boxes = []
    confidences = []
    class_ids = []

    # each layer outputs
    for output in layer_outputs:
      for detection in output:
        # get id and confidence
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # filter out low confidence
        if confidence > self.ct:
          # scale bounding box back to the image size
          box = detection[0:4] * np.array([width, height, width, height])
          (cx, cy, w, h) = box.astype('int')

          # grab the top-left corner of the box
          tx = int(cx - (w / 2))
          ty = int(cy - (h / 2))

          # check top-left corner of the box
          # if x is negative
          if (tx < 0):
            w = cx + (w / 2)
            cx = w / 2
          # if x is more than image width
          elif (tx > width):
            w = width - cx - (w / 2)
            cx = width - (w / 2)

          # if y is negative substitute to height/2 
          # if y is negative
          if (ty < 0):
            h = cy + (h / 2)
            cy = h / 2
          # if y is more than image height
          elif (ty > height):
            h = height - cy - (h / 2)
            cy = height - (h / 2)

          # update lists
          boxes.append([cx,cy,int(w),int(h)])
          confidences.append(float(confidence))
          class_ids.append(class_id)

    # apply NMS
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.ct, self.nmst)
    return boxes, class_ids, idxs

  # runs detection on the image
  def detect(self, img: np.ndarray) -> List:
    # get output
    layer_outputs = self.get_inf(img)

    # filter thresholds and target
    height, width = img.shape[:2]
    boxes, class_ids, idxs = self.thresh(layer_outputs, width, height)

    res = []
    # check for zero
    if len(idxs) > 0:
      # loop over indices
      for i in idxs.flatten():
        # extract detection data
        res.append({
          'class_id': class_ids[i],
          'x': boxes[i][0],
          'y': boxes[i][1],
          'w': boxes[i][2] if boxes[i][2] < width else width,
          'h': boxes[i][3] if boxes[i][3] < height else height
        })

    return res
