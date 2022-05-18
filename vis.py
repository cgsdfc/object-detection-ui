import os
from shutil import copyfile
import numpy as np

import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from joblib import Parallel, delayed
import cv2
from pathlib import Path as P


def draw_anchor_box(res_path,
                    output_path,
                    conf_thresh=0.3,
                    vis_classes=None,
                    verbose=False,
                    raw_images_path=None,
                    raw_images_list=None):
    """给输入的图像画上检测框，根据模型的输出结果。输入图像必须是jpg格式。

    raw_images_path：存有原始输入图像的路径。
    res_path：模型检测结果，格式参考：
    output_path：结果输出路径，图像命名和原始图像一致。
    conf_thresh: 置信度阈值，只有超过这个值才会被认为是合法的框。
    vis_classes: 要显示的类别，默认只会显示airplane和ship。
    """

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # 处理两种输入，list和path。
    assert (raw_images_path is None) ^ (raw_images_list is None)
    if raw_images_list is None:  # 支持直接输入路径列表。
        assert raw_images_path is not None
        raw_images_list = list(P(raw_images_path).glob('*.jpg'))
    else:
        raw_images_list = list(map(P, raw_images_list))

    result = []
    for p in raw_images_list:
        pnew = P(output_path).joinpath(p.name)
        # copyfile(str(p), str(pnew))
        result.append(pnew)

    if vis_classes is None:
        vis_classes = ["airplane", "ship"]
        color = {"airplane": (0, 215, 255), "ship": (48, 48, 255)}
    else:
        color = {cls: tuple(np.random.randint(low=0, high=256, size=[3])) for cls in vis_classes}

    if verbose:
        print(f'输出路径：{output_path}')

    vis_res_path = [os.path.join(res_path, res_name) for res_name in os.listdir(res_path) if
                    res_name.split(".")[0].split("_")[-1] in vis_classes]

    if verbose:
        print(f'模型关于类别的输出文件：{vis_res_path}')

    # 图像名字到上面所有锚框+置信度的映射
    images_to_boxes = defaultdict(list)

    for cls, path in zip(vis_classes, vis_res_path):
        if verbose:
            print(f'可视化类别：{cls} 模型输出：{path}')
        with open(path, "r") as res_file:
            lines = res_file.readlines()
            for line in lines:
                img_name, conf, x_min, y_min, x_max, y_max = line.split()
                img_name = img_name + ".jpg"
                x_min = int(float(x_min))
                x_max = int(float(x_max))
                y_min = int(float(y_min))
                y_max = int(float(y_max))
                conf = round(float(conf), 2)
                if conf < conf_thresh:
                    continue
                input_image = os.path.join(output_path, img_name)
                if not os.path.exists(input_image):
                    # 文件不存在，说明这个文件不需要可视化。
                    continue
                box = [x_min, y_min, x_max, y_max]
                images_to_boxes[img_name].append((box, cls, conf))

    for input_image in tqdm.tqdm(raw_images_list):
        img_name = input_image.name
        output_image = os.path.join(output_path, img_name)
        box_list = images_to_boxes[img_name]
        img = cv2.imread(str(input_image), cv2.IMREAD_COLOR)

        for box, cls, conf in box_list:
            [x_min, y_min, x_max, y_max] = box
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color[cls], thickness=2, lineType=8)
            cv2.putText(img, cls + " " + str(conf), (x_min - 5, y_min - 5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.75, color=color[cls], thickness=2, lineType=8)

        cv2.imwrite(output_image, img)

    return result, images_to_boxes


if __name__ == '__main__':
    raw_images_path = '/home/liao/codes/Object_Detection_UI/test_images/input'
    raw_images_list = list(P(raw_images_path).glob('*.jpg'))
    import json

    # json.dump(
    #     dict(raw_images_list=list(map(str, raw_images_list)),
    #          res_path='/home/liao/codes/Results/results/nwpu_p_30shot_novel0_neg0/ene000050',
    #          output_path='/home/liao/codes/Object_Detection_UI/test_images/output3', ),
    #     P('/test_images/tmp/input_json.json').open('w'),
    #     indent=4,
    # )

    result, images_to_boxes = draw_anchor_box(
        raw_images_list=raw_images_list,
        res_path='/home/liao/codes/Results/results/nwpu_p_30shot_novel0_neg0/ene000050',
        output_path='/home/liao/codes/Object_Detection_UI/test_images/tmp/output_images',
        verbose=True,
    )
    json.dump(
        dict(result=list(map(str, result)), images_to_boxes=images_to_boxes),
        P('./test_images/tmp/output_json.json').open('w'),
        indent=4,
    )
