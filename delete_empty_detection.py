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
                      type=str, default="data")
  arg = parser.parse_args()

  data_path = arg.data_path

  # loop every directory in <data_path> directory
  for name in os.listdir(data_path):
    target_dir = os.path.join(data_path, name)
    if os.path.isdir(target_dir):
      print(f"\nRunning on directory {target_dir}...")

      with open(f"{target_dir}.txt", "r") as file_list_text:

        for file in tqdm(os.listdir(target_dir)):
          full_path = os.path.join(target_dir, file)
          file_name = full_path.split('.')[0]
          extension = full_path.split('.')[-1]

          if os.path.isfile(full_path) and extension == 'txt':
            line_in_file_list = file_list_text.readline().strip()
            # check if file txt is empty -> no detection
            if os.stat(full_path).st_size == 0:
              # delete txt file
              os.remove(full_path)

              # delete image with different extension
              for image_extension in ['jpg', 'png', 'jpeg', 'tiff', 'bmp', 'gif']:
                try:
                  os.remove(file_name + '.' + image_extension)
                  break
                except:
                  print('file extensions do not match')
              print(f"deleted ({file_name})")
            else:
              with open(f"{target_dir}_revision.txt", "a") as file:
                # only write line when the data is not deleted
                file.write(line_in_file_list + "\n")
      
      os.remove(f"{target_dir}.txt")
      os.rename(f"{target_dir}_revision.txt", f"{target_dir}.txt")
