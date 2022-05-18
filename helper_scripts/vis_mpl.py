import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2 as cv


def bbox_to_rect(bbox, color):  # 本函数已保存在d2lzh包中方便以后使用
    # 将边界框(左上x, 左上y, 右下x, 右下y)格式转换成matplotlib格式：
    # ((左上x, 左上y), 宽, 高)
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
        fill=False, edgecolor=color, linewidth=2)


# data = pd.read_csv('dev.csv')
# print(data["label_path"].iloc[1])

boxes = []
labels = []
areas = []
iscrowd = []
with open(data['label_path'][4]) as f:
    for line in f.readlines()[1:]:
        temp = line.strip().split(' ')
        xmin = int(temp[1])
        ymin = int(temp[2])
        xmax = int(temp[3])
        ymax = int(temp[4])
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(int(temp[0]))
        areas.append((xmax - xmin) * (ymax - ymin))
        iscrowd.append(0)

image = cv.imread(data['img_path'][4])
fig = plt.imshow(image)
for i, box in enumerate(boxes):
    rect = bbox_to_rect(box, 'red')
    fig.axes.add_patch(rect)
    fig.axes.text(rect.xy[0] + 24, rect.xy[1] + 10, "pedestrian",
                  va='center', ha='center', fontsize=6, color='blue',
                  bbox=dict(facecolor='m', lw=0))
plt.show()
