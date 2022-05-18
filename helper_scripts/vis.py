import cv2
import os
from shutil import copyfile

import tqdm

conf_thresh = 0.3
raw_images_path = "测图片路径"
res_path = "检测结果路径"
output_path = "可时化后保存路径"
if not os.path.exists(output_path):
    os.makedirs(output_path)
for name in [name for name in os.listdir(raw_images_path) if name.endswith(".jpg")]:
    copyfile(os.path.join(raw_images_path, name), os.path.join(output_path, name))

vis_classes = ["airplane", "ship"]
color = {"airplane": (0, 215, 255), "ship": (48, 48, 255)}
vis_res_path = [os.path.join(res_path, res_name) for res_name in os.listdir(res_path) if
                res_name.split(".")[0].split("_")[-1] in vis_classes]
print(vis_res_path)

for cls, path in zip(vis_classes, vis_res_path):
    print("vis", cls)
    with open(path, "r") as res_file:
        lines = res_file.readlines()
        for line in tqdm.tqdm(lines):
            img_name, conf, x_min, y_min, x_max, y_max = line.split()
            img_name = img_name + ".jpg"
            x_min = int(float(x_min))
            x_max = int(float(x_max))
            y_min = int(float(y_min))
            y_max = int(float(y_max))
            conf = round(float(conf), 2)
            if conf < conf_thresh:
                continue
            # print("vis "+cls,img_name, conf, x_min, y_min, x_max, y_max)
            img = cv2.imread(os.path.join(output_path, img_name), cv2.IMREAD_COLOR)

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color[cls], thickness=2, lineType=8)
            cv2.putText(img, cls + " " + str(conf), (x_min - 5, y_min - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.75, color=color[cls], thickness=2, lineType=8)
            cv2.imwrite(os.path.join(output_path, img_name), img)
