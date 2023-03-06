# Object Detection UI

This is a multi-panel visualization tool for training and inference of an object detection model (e.g., the YOLO family, SSD, etc). It has the following panels:
1. Configuration panel, where you set the paths of different configuration files.
2. Training visualization panel, where the model is trained in the background with its loss and accuracy plotted in the frame (left-hand side) and logged in the console (right-hand side).
3. Single image detection panel, where an input image and the detection visualization of it are displayed side by side for comparison.
4. Multiple image detection panel, where multiple images are detected in a single batch with the results replacing them.


## Screenshots

Configuration panel:
<div align=left>
<img src="screenshots/四大面板/配置面板.png" width="500">
</div>

Training visualization panel:
<div align=left>
<img src="screenshots/训练面板/训练-训练过程中.png" width="500">
</div>

Single image detection panel:
<div align=left>
<img src="screenshots/单图面板/单图-已识别.png" width="500">
</div>

Multiple image detection panel:
<div align=left>
<img src="screenshots/批量面板/批量-检测进行中.png" width="500">
</div>


## Usage

1. Install dependencies with `requirements.txt`.
2. Run `main.py`.
