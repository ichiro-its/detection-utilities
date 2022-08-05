#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <map>
#include <fstream>
#include <string>
#include <vector>

class DNNObjectDetector
{
public:

  int label;
  float score;
  float left;
  float top;
  float width;
  float height;
};

class DNNDetector
{
public:

  enum
  { 
    OBJECT_TYPE_BALL            = 0,
    OBJECT_TYPE_GOALPOST        = 1,
    OBJECT_TYPE_ROBOT           = 2,
    OBJECT_TYPE_L_INTERSECTION  = 3,
    OBJECT_TYPE_T_INTERSECTION  = 4,
    OBJECT_TYPE_X_INTERSECTION  = 5
  };

  DNNDetector();

  void setSize(int size);
  void loadModel(std::string class_file_path, std::string model_configuration_path, std::string model_weights_path);
  std::vector<DNNObjectDetector> detect(cv::Mat image);

private:

  std::vector<std::string> classes;
  std::vector<cv::Mat> outs;
  std::vector<DNNObjectDetector> detection_results;

  cv::dnn::Net net;

  std::string class_file_path;
  std::string model_configuration_path;
  std::string model_weights_path;

  float conf_threshold;
  float nms_threshold;

  double img_width;
  double img_height;

  int size;
};

DNNDetector::DNNDetector()
{
  this->size = 416;
  this->conf_threshold = 0.4;
  this->nms_threshold = 0.3;
}

void DNNDetector::setSize(int size) {
  this->size = size;
}

void DNNDetector::loadModel(std::string class_file_path, std::string model_weights_path, std::string model_configuration_path)
{
  this->class_file_path = class_file_path;
  this->model_configuration_path = model_configuration_path;
  this->model_weights_path = model_weights_path;

  net = cv::dnn::readNet(model_weights_path, model_configuration_path, "");
  // net = cv::dnn::readNetFromONNX(model_weights_path);

  std::ifstream ifs(class_file_path.c_str());
  std::string line;
  while (std::getline(ifs, line)) {
    classes.push_back(line);
  }

  // Set computation method
  net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
  // net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);

  //net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
  // net.setPreferableTarget(cv::dnn::DNN_TARGET_MYRIAD);
  net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL_FP16);
}

std::vector<DNNObjectDetector> DNNDetector::detect(cv::Mat image)
{
  std::vector<DNNObjectDetector> detection_results;
  std::vector<cv::String> layer_output = net.getUnconnectedOutLayersNames();

  // Create a 4D blob from a frame
  static cv::Mat blob;
  cv::Size input_size = cv::Size(size, size);
  cv::dnn::blobFromImage(image, blob, 1.0, input_size, cv::Scalar(), false, false, CV_8U);

  net.setInput(blob, "", 0.00392, cv::Scalar(0, 0, 0, 0));
  net.forward(outs, layer_output);

  // Get width and height from image
  img_width = static_cast<double>(image.cols);
  img_height = static_cast<double>(image.rows);

  static std::vector<int> out_layers = net.getUnconnectedOutLayers();
  static std::string out_layer_type = net.getLayer(out_layers[0])->type;

  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  if (out_layer_type == "Region") {
    for (size_t i = 0; i < outs.size(); ++i) {
      // Network produces output blob with a shape NxC where N is a number of
      // detected objects and C is a number of classes + 4 where the first 4
      // numbers are [center_x, center_y, width, height]
      float * data = reinterpret_cast<float *>(outs[i].data);
      for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
        cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
        cv::Point class_id_point;
        double confidence;
        cv::minMaxLoc(scores, 0, &confidence, 0, &class_id_point);
        if (confidence > conf_threshold) {
          double centerX = data[0] * img_width;
          double centerY = data[1] * img_height;
          double width = data[2] * img_width;
          double height = data[3] * img_height;
          double left = centerX - width / 2;
          double top = centerY - height / 2;

          class_ids.push_back(class_id_point.x);
          confidences.push_back(static_cast<float>(confidence));
          boxes.push_back(cv::Rect(left, top, width, height));
        }
      }
    }
  }

  // NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends
  // we need NMS in sample or NMS is required if number of outputs > 1
  if (out_layers.size() > 1) {
    std::map<int, std::vector<size_t>> class2indices;
    for (size_t i = 0; i < class_ids.size(); i++) {
      if (confidences[i] >= conf_threshold) {
        class2indices[class_ids[i]].push_back(i);
      }
    }

    std::vector<cv::Rect> nms_boxes;
    std::vector<float> nms_confidences;
    std::vector<int> nms_class_ids;
    for (std::map<int, std::vector<size_t>>::iterator it = class2indices.begin();
      it != class2indices.end(); ++it)
    {
      std::vector<cv::Rect> local_boxes;
      std::vector<float> local_confidences;
      std::vector<size_t> class_indices = it->second;
      for (size_t i = 0; i < class_indices.size(); i++) {
        local_boxes.push_back(boxes[class_indices[i]]);
        local_confidences.push_back(confidences[class_indices[i]]);
      }
      std::vector<int> nms_indices;
      cv::dnn::NMSBoxes(local_boxes, local_confidences, conf_threshold, nms_threshold, nms_indices);
      for (size_t i = 0; i < nms_indices.size(); i++) {
        size_t idx = nms_indices[i];
        nms_boxes.push_back(local_boxes[idx]);
        nms_confidences.push_back(local_confidences[idx]);
        nms_class_ids.push_back(it->first);
      }
    }
    boxes = nms_boxes;
    class_ids = nms_class_ids;
    confidences = nms_confidences;
  }

  if (boxes.size()) {
    for (size_t i = 0; i < boxes.size(); ++i) {
      cv::Rect box = boxes[i];
      if (box.width * box.height != 0) {
        // Add detected object into vector
        DNNObjectDetector detection_object;

        detection_object.label = class_ids[i];
        detection_object.score = confidences[i];
        detection_object.left = box.x;
        detection_object.top = box.y;
        detection_object.width = box.width;
        detection_object.height = box.height;

        detection_results.push_back(detection_object);
      }
    }
  }

  return detection_results;
}

int main() {
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;

  DNNDetector detector = DNNDetector();
  detector.loadModel("data/obj.names", "data/yolo_weights.weights", "data/config.cfg");

  std::vector<int> input_sizes = {192, 224, 320, 416};

  std::vector<std::string> filenames;
  cv::glob("data", filenames);
  std::string image_extension[] = {"png", "jpg", "jpeg", "tiff", "bmp", "gif"};

  std::cout << "=====YOLO TEST=====\n";

  for (auto input_size: input_sizes) {
    double ms_double = 0.0;
    detector.setSize(input_size);
    for (size_t i = 0; i < filenames.size(); i++)
    {
      //store the position of last '.' in the file name
      int position=filenames[i].find_last_of(".");
      //store the characters after the '.' from the file_name string
      std::string file_extension = filenames[i].substr(position+1);
      // check if the file is image
      bool exists = std::count(std::begin(image_extension), std::end(image_extension), file_extension) > 0;

      if (exists) {
        cv::Mat image = cv::imread(filenames[i]);

        auto t1 = high_resolution_clock::now();
        auto start_time = std::chrono::system_clock::now();
        std::vector<DNNObjectDetector> detection_results = detector.detect(image);

        if (i > 0) {
          auto t2 = high_resolution_clock::now();
          ms_double += std::chrono::duration<double, std::milli>(t2 - t1).count();
        }
      }
    }

    std::cout << "Input size: " << input_size << " | Average latency: " << ms_double / (filenames.size() - 1) << " ms" << std::endl;

  }

  return 0;
}
