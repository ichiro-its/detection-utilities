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
import os
from tqdm import tqdm

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', help='path for image data',
                      type=str, default="data_coba")
  arg = parser.parse_args()

  data_path = arg.data_path

  # loop every directory in <data_path> directory
  for name in os.listdir(data_path):
    target_dir = os.path.join(data_path, name)
    if os.path.isdir(target_dir):
      print(f"\nRunning on directory {target_dir}...")


      for everything in tqdm(os.listdir(target_dir)):
        full_path = os.path.join(target_dir, everything)
        file_name = full_path.split('.')[0]
        extension = full_path.split('.')[-1]

        if os.path.isfile(full_path) and extension == 'txt':
          with open(f"{file_name}.txt", "r") as file:
            detected_objects = file.readlines()
          
          for i in range(len(detected_objects)):
            class_id, center_x, center_y, w, h = detected_objects[i].split()

            w = float(w)
            h = float(h)
            x = float(center_x) - w/2
            y = float(center_y) - h/2
            detected_objects[i] = f"{class_id} {x} {y} {w} {h}\n"
          
          with open(f"{file_name}.txt", "w") as file:
            file.writelines(detected_objects)


