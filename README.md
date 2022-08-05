# detection-utilities
This repository contains utilities to perform Pseudo Labeling. 

## Scripts

### Perform Pseudo Label
For now, our script just create YOLO labels that saved in txt file.
There are some optional arguments that we can add when run the script, including --data_path, --CUDA, etc.

`python3 pseudo_label.py`

### Delete Related Files when the Detection Result is Empty
To make our dataset cleaner, our script can delete annotation file and image when the detection result is empty. 
It can also delete line in the file that contain all the image's name.
In this script there is just one optional argument (--data_path), the default value is "data". 

`python3 delete_empty_detection.py`

### Change Annotation Format
YOLO annotation's format that we want is `(class_id, x, y, width, height)`. This script is used to convert `(class_id, center_x, center_y, width, height)` format to our format.

`python3 change_annotation_format.py`

### Check YOLO model Latency
```
g++ check_latency.cpp -o check_latency.o `pkg-config --cflags --libs opencv4`
./check_latency.o
```
