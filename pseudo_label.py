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

import argparse
import cv2
import os
from tqdm import tqdm

from yolo import Yolo

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--CONF_THRESH', help='threshold for object confident',
                      type=float, default=0.4)
  parser.add_argument('--config', help='path for yolo config',
                      type=str, default="data/config.cfg")
  parser.add_argument('--CUDA', help='to enable or disable CUDA',
                      type=bool, default=False)
  parser.add_argument('--data_path', help='path for image data',
                      type=str, default="data")
  parser.add_argument('--draw', help='draw detection result on image',
                      type=bool, default=True)               
  parser.add_argument('--labels', help='path for yolo label',
                      type=str, default="data/obj.names")
  parser.add_argument('--NMS_THRESH', help='threshold for Non-maximum Suppression',
                      type=float, default=0.4)
  parser.add_argument('--weights', help='path for yolo weight',
                      type=str, default="data/yolo_weights.weights")  
  arg = parser.parse_args()

  data_path = arg.data_path
  weights = arg.weights
  config = arg.config
  labels = arg.labels
  draw_output = arg.draw

  # init yolo network
  conf_thresh = arg.CONF_THRESH # less == more boxes (but more false positives)
  nms_thresh = arg.NMS_THRESH # less == more boxes (but more overlap)
  net = Yolo(config, weights, labels, conf_thresh, nms_thresh, use_cuda=arg.CUDA)

  for name in os.listdir(data_path):
    target_dir = os.path.join(data_path, name)
    if os.path.isdir(target_dir):
      print(f"\nRunning on directory {target_dir}...")

      file_list_text = open(f"{target_dir}.txt", "w")

      if draw_output:
        output_dir = os.path.join(target_dir, "output")
        os.makedirs(output_dir,  exist_ok=True)

      for image_name in tqdm(os.listdir(target_dir)):
        full_path = os.path.join(target_dir, image_name)
        if os.path.isfile(full_path) and full_path.split('.')[-1] in ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif']:
          file_list_text.write(full_path + "\n")

          image = cv2.imread(full_path)
          height, width = image.shape[:2]
          detected_objects = net.detect(image)

          image_text = open(f"{full_path.split('.')[0]}.txt", "w")
          for detected_object in detected_objects:
            image_text.write(
              f"{detected_object['class_id']} {detected_object['x'] / width} "
              f"{detected_object['y'] / height} {detected_object['w'] / width} "
              f"{detected_object['h'] / height}\n"
            )
          image_text.close()

          if draw_output:
            colors = [(255,0,0), (0,255,0), (0,0,255), (0,255,255), (255,0,255), (255,255,0)]

            for detected_object in detected_objects:
              x = int(detected_object['x'] - detected_object['w'] / 2)
              y = int(detected_object['y'] - detected_object['h'] / 2)
              x2 = int(detected_object['x'] + detected_object['w'] / 2)
              y2 = int(detected_object['y'] + detected_object['h'] / 2)
              cv2.rectangle(image, (x, y),
                (x2, y2),
                colors[detected_object['class_id'] % len(colors)])

              cv2.putText(image, str(detected_object['class_id']), (int(detected_object['x']), int(detected_object['y'])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[detected_object['class_id']], 2)

            cv2.imwrite(os.path.join(output_dir, image_name), image)

      file_list_text.close()
