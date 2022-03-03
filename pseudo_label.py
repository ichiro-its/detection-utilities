import cv2
import os
from tqdm import tqdm

from yolo import Yolo

if __name__ == "__main__":
  data_path = "data"
  weights = os.path.join(data_path, "yolo_weights.weights")
  config = os.path.join(data_path, "config.cfg")
  labels = os.path.join(data_path, "obj.names")

  # init yolo network
  conf_thresh = 0.4 # less == more boxes (but more false positives)
  nms_thresh = 0.4 # less == more boxes (but more overlap)
  net = Yolo(config, weights, labels, conf_thresh, nms_thresh, use_cuda=False)

  for name in os.listdir(data_path):
    target_dir = os.path.join(data_path, name)
    if os.path.isdir(target_dir):
      print(f"\nRunning on directory {target_dir}...")

      file_list_text = open(f"{target_dir}.txt", "w")

      for image in tqdm(os.listdir(target_dir)):
        full_path = os.path.join(target_dir, image)
        if os.path.isfile(full_path) and full_path.split('.')[-1] in ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif']:
          file_list_text.write(f"{full_path}.txt\n")
          file_list_text.write(full_path)
          detected_objects = net.detect(cv2.imread(full_path))

          image_text = open(f"{full_path.split('.')[0]}.txt", "w")
          for detected_object in detected_objects:
            image_text.write(f"{detected_object['class_id']} {detected_object['x']} {detected_object['y']} {detected_object['w']} {detected_object['h']}\n")
          image_text.close()

      file_list_text.close()
