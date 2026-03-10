from ultralytics import YOLO


class YoloDetector:
  def __init__(self, model_path, confidence, class_list=None):
    self.model = YOLO(model_path)
    # If class_list is None, allow all classes detected by the model.
    # Otherwise, only keep detections whose class name is in the list.
    self.classList = class_list
    self.confidence = confidence

  def detect(self, image):
    results = self.model.predict(image, conf=self.confidence)
    result = results[0]
    detections = self.make_detections(result)
    return detections

  def make_detections(self, result):
    boxes = result.boxes
    detections = []
    for box in boxes:
      x1, y1, x2, y2 = box.xyxy[0]
      x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
      w, h = x2 - x1, y2 - y1
      class_number = int(box.cls[0])
      class_name = result.names[class_number]
      if self.classList is not None and class_name not in self.classList:
        continue
      conf = float(box.conf[0])
      detections.append((([x1, y1, w, h]), conf, class_number))
    return detections
