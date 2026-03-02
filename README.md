<h1 align="center">Bee Behavior Tracking System</h1>

<p align="center">
  This project uses computer vision to track and quantify the movement of 5–10 bees inside an artificial hive.
</p>

---

<h2 id="overview">Overview</h2>
<p>
  By combining YOLO-based object detection with multi-object tracking (Deep SORT / similar), the system generates behavioral metrics such as speed and activity levels.
  The purpose is to compare control and treatment groups to determine whether engineered gut microbiome modifications produce measurable changes in bee behavior.
</p>

---

<h2 id="credits">Credits & Origin</h2>
<p>
  This project is adapted from the YOLO + Deep SORT example by
  <a href="https://github.com/iamrukeshduwal/yolov11_real_time_object_detection_with_DeepSORT">
    yolov11_real_time_object_detection_with_DeepSORT
  </a>.
  The original repository provides the core YOLO 11 detection and Deep SORT tracking pipeline; this fork repurposes it for bee behavior tracking experiments.
</p>

---

<h2 id="usage">Usage</h2>
<p>
  The basic usage is the same as in the original project:
</p>

<pre><code>pip install -r requirements.txt
python yolo_detection_tracker.py
</code></pre>

<p>
  Replace the example video with footage of your artificial hive, and adjust any experiment-specific parameters (such as regions of interest, confidence thresholds, or bee-specific class filtering) inside <code>yolo_detection_tracker.py</code>.
</p>

---

<h2 id="license">License</h2>
<p>
  This repository follows the licensing terms of the original project by
  <a href="https://github.com/iamrukeshduwal">Rukesh Duwal</a>. Please see the upstream repository
  for more details and ensure that any derivative work properly credits the original author.
</p>
