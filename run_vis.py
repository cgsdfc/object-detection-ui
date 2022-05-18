import vis
import argparse
import json
import sys


def input_json_schema(json):
    try:
        assert isinstance(json['res_path'], str)
        assert isinstance(json['output_path'], str)
        assert isinstance(json['raw_images_list'], list)
    except Exception:
        raise ValueError(f'输入json格式错误')


def output_json_schema(result, images_to_boxes):
    return dict(
        result=result,
        images_to_boxes=dict(images_to_boxes),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 输入是一个json文件。
    parser.add_argument('input_json', help='输入一个json文件，指定参数')
    args = parser.parse_args()
    with open(args.input_json) as f:
        input_json = json.load(f)

    input_json_schema(input_json)
    result, image_to_boxes = vis.draw_anchor_box(**input_json)
    result = list(map(str, result))
    output_json = output_json_schema(result, image_to_boxes)
    json.dump(output_json, sys.stdout, indent=4)
