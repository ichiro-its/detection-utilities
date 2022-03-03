import cv2
import os
from tqdm import tqdm

from yolo import Yolo

if __name__ == "__main__":
  # TODO: change to args
  data_path = "data"
  weights = os.path.join(data_path, "yolo_weights.weights")
  config = os.path.join(data_path, "config.cfg")
  labels = os.path.join(data_path, "obj.names")
  draw_output = True

  # init yolo network
  conf_thresh = 0.4 # less == more boxes (but more false positives)
  nms_thresh = 0.4 # less == more boxes (but more overlap)
  net = Yolo(config, weights, labels, conf_thresh, nms_thresh, use_cuda=False)

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
              cv2.rectangle(image, (detected_object['x'], detected_object['y']),
                (detected_object['x'] + detected_object['w'], detected_object['y'] + detected_object['h']),
                colors[detected_object['class_id'] % len(colors)])

              cv2.putText(image, str(detected_object['class_id']), (detected_object['x'] + 3, detected_object['y'] + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[detected_object['class_id']], 2)

            cv2.imwrite(os.path.join(output_dir, image_name), image)

      file_list_text.close()
