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

          # update lists
          boxes.append([tx,ty,int(w),int(h)])
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
          'x': boxes[i][0] if boxes[i][0] > 0 else 0,
          'y': boxes[i][1] if boxes[i][1] > 0 else 0,
          'w': boxes[i][2] if boxes[i][2] < width else width,
          'h': boxes[i][3] if boxes[i][3] < height else height
        })

    return res
