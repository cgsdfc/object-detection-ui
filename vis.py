import cv2
import os
from shutil import copyfile
import numpy as np

import tqdm


# conf_thresh=0.3
# raw_images_path = "测图片路径"
# res_path = "检测结果路径"
# output_path = "可时化后保存路径"


def draw_anchor_box(raw_images_path, res_path, output_path, conf_thresh=0.3, vis_classes=None, verbose=False):
    """给输入的图像画上检测框，根据模型的输出结果。输入图像必须是jpg格式。

    raw_images_path：存有原始输入图像的路径。
    res_path：模型检测结果，格式参考：
    output_path：结果输出路径，图像命名和原始图像一致。
    conf_thresh: 置信度阈值，只有超过这个值才会被认为是合法的框。
    vis_classes: 要显示的类别，默认只会显示airplane和ship。
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    if vis_classes is None:
        vis_classes = ["airplane", "ship"]
        color = {"airplane": (0, 215, 255), "ship": (48, 48, 255)}
    else:
        color = {cls: tuple(np.random.randint(low=0, high=256, size=[3])) for cls in vis_classes}

    print(f'可视化类别：{vis_classes}')
    print(f'类别的颜色：{color}')

    vis_res_path = [os.path.join(res_path, res_name) for res_name in os.listdir(res_path) if
                    res_name.split(".")[0].split("_")[-1] in vis_classes]
    print(f'模型关于类别的输出文件：{vis_res_path}')

    for cls, path in zip(vis_classes, vis_res_path):
        print(f'可视化类别：{cls} 模型输出：{path}')

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

                input_image = os.path.join(raw_images_path, img_name)
                if not os.path.isfile(input_image):
                    raise ValueError(f'文件不存在：{input_image}')

                output_image = os.path.join(output_path, img_name)
                if verbose:
                    print(f'输入图像：{input_image}')
                    print(f'输出图像：{output_image}')

                img = cv2.imread(input_image, cv2.IMREAD_COLOR)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color[cls], thickness=2, lineType=8)
                cv2.putText(img, cls + " " + str(conf), (x_min - 5, y_min - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.75, color=color[cls], thickness=2, lineType=8)
                cv2.imwrite(output_image, img)


if __name__ == '__main__':
    draw_anchor_box(
        raw_images_path='/home/liao/codes/FSODM/dataset/NWPU/evaluation/images',
        res_path='/home/liao/codes/Results/results/nwpu_p_10shot_novel0_neg0/ene000050',
        output_path='/home/liao/codes/Object_Detection_UI/test_images/output2'
    )