import enum
import os
import sys
import random
import shlex
import signal
import subprocess
import time
import json
import traceback
from datetime import datetime

from collections import defaultdict
from pathlib import Path as P

# import vis
import pyqtgraph as pg
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import (
    QMessageBox,
    QFileDialog,
    QPushButton,
    QLineEdit,
    QLabel,
    QProgressBar,
)
from pyqtgraph import PlotDataItem, PlotItem

from Ui_develop import *

import numpy as np

MOCKED_TRAIN = False
MOCKED_OBJDET = False
print(f'MOCKED_TRAIN={MOCKED_TRAIN}')
print(f'MOCKED_OBJDET={MOCKED_OBJDET}')

CODES_DIR = "/home/liao/codes"

# 目标检测结果的预生成目录。
MOCKED_IMAGE_RESULT_DIR = P("/home/liao/codes/Results/vis/conf0-0.5/0.5/nwpu_p_30shot_novel0_neg0")

THIS_PROJECT_DIR = P(__file__).parent


# print(f'UI项目路径：{THIS_PROJECT_DIR}')


def output_image_dir():
    "预先挑选的检测目录"
    if MOCKED_OBJDET:
        return THIS_PROJECT_DIR / 'test_images' / 'output'
    return None


def input_image_dir():
    "选择输入图像的目录"
    # 必须返回str，因为是给Qt接口用的。
    return str(THIS_PROJECT_DIR / 'test_images' / 'input')


def detection_result_dir():
    # res_path = '/home/liao/codes/Results/results/nwpu_p_10shot_novel0_neg0/ene000050'
    res_path = THIS_PROJECT_DIR / 'test_images' / 'detection_result'
    return P(res_path)


def train_log_paths():
    "用来mocked的训练日志"
    train_log = THIS_PROJECT_DIR / 'train_log'
    return {
        TrainMode.pretrain: train_log / 'train_loss.txt',
        TrainMode.fewtune: train_log / 'fewtune_loss.txt',
    }


def export_dir():
    dir = THIS_PROJECT_DIR.joinpath('export')
    dir.mkdir(exist_ok=True)
    return str(dir)


def python_dir():
    "选择Python解析器的备选路径"
    return "/home/liao/anaconda3/envs"


def objdet_python():
    return '/home/liao/anaconda3/bin/python'


class StringConstants:
    "字符串常量，防止编码错误"
    precision = "precision"
    recall = "recall"
    recall50 = "recall50"
    total = "total"
    loss = "loss"
    acc = "acc"  # 测试用
    # objdet
    images_to_boxes = 'images_to_boxes'
    raw_images_list = 'raw_images_list'
    res_path = 'res_path'
    output_path = 'output_path'
    result = 'result'
    input_json_dot_json = 'input_json.json'


class LoglineParseResult:
    "训练日志行的原始解析结果，字段"
    precision = None
    recall50 = None
    recall75 = None
    cls_acc = None
    loss_x = None
    loss_y = None
    loss_w = None
    loss_h = None
    cls = None
    total = None
    conf = None


# 将训练日志的字段和我们想要的字段做映射。
METRICS_MAP = {
    StringConstants.loss: StringConstants.total,
    StringConstants.recall: StringConstants.recall50,
    StringConstants.precision: StringConstants.precision,
}


def parse_logline(line: str, map=None):
    "解析一行训练日志，若map存在，将获取特定字段并且重命名。"
    line = line.split(", ")
    ans = LoglineParseResult()
    ans.precision = float(line[1].split()[-1])
    ans.recall50 = float(line[2].split()[-1])
    ans.recall75 = float(line[3].split()[-1])
    ans.cls_acc = float(line[4].split()[-1])

    ans.loss_x = float(line[5].split()[-1])
    ans.loss_y = float(line[6].split()[-1])
    ans.loss_w = float(line[7].split()[-1])
    ans.loss_h = float(line[8].split()[-1])

    ans.conf = float(line[9].split()[-1])
    ans.cls = float(line[10].split()[-1])
    ans.total = float(line[11].split()[-1])
    if map is None:
        return ans
    # 把想要的字段改个名字扣出来。
    ans = {key: getattr(ans, value) for key, value in map.items()}
    return ans


class TrainMode(enum.Enum):
    "训练模式的enum版本"
    pretrain = 0
    fewtune = 1


# UI上的字符串映射到训练模式。
TEXT_TO_TRAIN_MODE = {
    "预训练": TrainMode.pretrain,
    "小样本微调": TrainMode.fewtune,
}


def random_bool():
    return random.random() > 0.5


def is_train_step_line(line: str):
    return "nGT" in line


class TrainThreadBase(QThread):
    # 拉起训练进程后发送此信号，反馈【是否拉起成功】
    train_start_signal = pyqtSignal(bool)
    # 收到训练进程的一行日志后发送此信号，传输【训练日志】
    train_step_signal = pyqtSignal(str)
    # 训练进程退出后发送此信号，反馈【进程退出状态码】
    train_end_signal = pyqtSignal(int)
    # 收到调用方的中断请求，并且处理完毕后，发送此信号。
    train_interrupt_signal = pyqtSignal(bool)


class TrainThreadMocked(TrainThreadBase):
    "Mocked 版训练线程，提供测试UI的数据"

    def __init__(self, cmd: list, cwd: str, mode: TrainMode) -> None:
        super().__init__()
        self.cmd = cmd
        self.cwd = cwd
        self.logfile = train_log_paths()[mode]
        print(f'读取日志文件：{self.logfile}')
        self.log_lines = P(self.logfile).read_text().splitlines()
        print(self.__class__.__name__)

    def run(self):
        print("训练线程启动")
        start_ok = random_bool()
        self.train_start_signal.emit(start_ok)
        if not start_ok:
            print(f'训练进程启动失败')
            return

        for epoch, line in enumerate(self.log_lines):
            if self.isInterruptionRequested():
                print(f"调用方请求中断")
                self.train_interrupt_signal.emit(random_bool())
                return
            if is_train_step_line(line):
                self.msleep(1000)  # 模拟一个训练步的耗时。
            print("训练线程发送日志")
            self.train_step_signal.emit(line)

        print("训练线程结束")
        rc = int(random_bool())  # 测试异常退出的情况。
        self.train_end_signal.emit(rc)


class TrainThread(TrainThreadBase):
    "训练线程的实现，负责管理一个训练进程的启动、抓取日志、中断清理，作为UI线程和训练进程之间的通信桥梁。"

    def __init__(self, cmd: list, cwd: str, mode: TrainMode) -> None:
        "cmd：要运行的命令；cwd：运行命令所在的目录。"
        super().__init__()
        if isinstance(cmd, str):
            cmd = shlex.split(cmd)
        self.cmd = cmd
        assert os.path.isdir(cwd)
        self.cwd = cwd
        # 到run的时候才拉起进程。
        self.p: subprocess.Popen = None  # 进程实例

    def handle_interrupt(self):
        "调用方向我发起中断请求。要杀死训练进程，并且保证其清理掉子进程，否则会把显存用完。返回是否进行了一次中断处理。"
        if not self.isInterruptionRequested():
            return False
        status = True
        if self.p is not None:
            print(f"正在杀死进程：{self.p.pid}")
            try:
                # 注意：必须用 KeyboardInterrupt 来杀死进程，即 SIGINT 2
                # 他才会做清理，如果是一般的kill（SIGKILL），他是不做任何清理的。
                # p.kill() p.terminate() 都是不行的。
                self.p.send_signal(signal.SIGINT)
                self.p.wait()
            except:
                status = False
                print(f"杀死进程时抛出异常：")
                traceback.print_exc()
            else:
                print(f"进程杀死成功")

        # 只要收到了中断请求，就必须发回响应。
        self.train_interrupt_signal.emit(status)
        return True

    def start_process(self):
        assert self.p is None

    def run(self):
        try:
            # 注意，必须把p赋值给self，否则无法杀死进程。
            self.p = p = subprocess.Popen(
                self.cmd,
                shell=False,  # shell=True 则cmd可以是str，否则必须是list
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.cwd,
            )
        except:
            traceback.print_exc()
            start_ok = False
            return
        else:
            start_ok = True
        finally:
            self.train_start_signal.emit(start_ok)

        self.handle_interrupt()

        while p.poll() is None:
            if self.handle_interrupt():
                return
            line = p.stdout.readline().strip()
            if line:
                line = line.decode()
                self.train_step_signal.emit(line)

        # 判断返回码状态
        print(f"训练线程：进程返回值：{p.returncode}")
        self.train_end_signal.emit(p.returncode)


class ObjdetThread(QThread):
    "Mocked 的目标检测后台线程"
    detection_error_signal = pyqtSignal()  # 是否正常启动了
    detection_result_signal = pyqtSignal(dict)  # 返回输出图像的路径的列表（前提是正常启动了）

    def __init__(self, input_image_list: list):
        super().__init__()
        input_image_list = list(map(P, input_image_list))
        self.input_image_list = input_image_list
        if MOCKED_OBJDET:
            self.run = self.run_copy_output
        else:
            self.run = self.run_vis
        self.working_dir = THIS_PROJECT_DIR / 'test_images' / 'tmp'
        self.working_dir.mkdir(exist_ok=True)

    def check_input(self):
        if not all(map(P.exists, self.input_image_list)):
            self.detection_error_signal.emit()

    def run_copy_output(self):
        "把预先生成的图像作为输出"
        self.check_input()
        if not output_image_dir().is_dir():
            self.detection_error_signal.emit()
            print(f'输出图像目录不存在：{output_image_dir()}')
            return
        result = []
        for input_image in self.input_image_list:
            output_image_path = output_image_dir().joinpath(input_image.name)
            if not output_image_path.exists():
                print(f'输出图像不存在：{output_image_path}')
                self.detection_error_signal.emit()
                return
            result.append(output_image_path)

        self.detection_result_signal.emit({StringConstants.result: result,
                                           StringConstants.images_to_boxes: None})

    def run_vis(self):
        "把模型的输出文本文件结合输入图像进行可视化后输出"
        self.check_input()
        if not detection_result_dir().is_dir():
            self.detection_error_signal.emit()
            print(f'模型预测输出目录不存在：{detection_result_dir()}')
            return
        input_json = {
            StringConstants.raw_images_list: list(map(str, self.input_image_list)),
            StringConstants.res_path: str(detection_result_dir()),
            StringConstants.output_path: str(self.working_dir.joinpath('output_images')),
        }
        input_json_file = self.working_dir.joinpath(StringConstants.input_json_dot_json)
        with input_json_file.open('w') as f:
            json.dump(input_json, f, indent=4)
        cmdline = [
            objdet_python(),
            THIS_PROJECT_DIR.joinpath('run_vis.py'),
            input_json_file,
        ]
        cmdline = list(map(str, cmdline))
        try:
            json_string = subprocess.check_output(cmdline).decode(encoding='utf8')
            output_json = json.loads(json_string)
        except:
            self.detection_error_signal.emit()
            print(f'调用脚本失败：')
            traceback.print_exc()
            return
        else:
            self.detection_result_signal.emit(output_json)


class ObjdetImagePanel:
    "多图检测的一个图片面板。"

    def __init__(self, label: QLabel) -> None:
        self.label = label
        self.clear()

    def clear(self):
        self.label.clear()
        self.input_image = None
        self.output_image = None
        self.box_list = None


class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        if MOCKED_TRAIN:
            self.TrainThreadClass = TrainThreadMocked
        else:
            self.TrainThreadClass = TrainThread

        # 辅助变量
        self.python_path = self.ui.le_python_path_pretrain.text
        self.reweight_config = self.ui.le_reweight_config.text
        self.backbone_config = self.ui.le_backbone_config.text
        self.pretrain_data_config = self.ui.le_data_config_pretrain.text
        self.fewtune_data_config = self.ui.le_data_config_fewtune.text
        self.fewtune_model_path = self.ui.le_model_path_fewtune.text
        self.console = self.ui.te_train_logging
        self.console.setReadOnly(True)
        self.train_mode_raw = self.ui.comboBox_train_mode.currentText

        # 配置面板
        self.init_config_tab()
        # 训练面板
        self.init_train_tab()
        # 目标检测
        self.init_objdet_tab()
        # 批量检测
        self.init_batchdet_tab()

    def update_interval(self):
        data = self.ui.le_update_interval.text()
        val = int(data)
        if not val:
            val = 5
        return val

    def init_train_tab(self):
        "训练面板的初始化"
        # 监控训练模式的变化。
        self.ui.comboBox_train_mode.currentTextChanged.connect(
            lambda: print(f"训练模式改变：{self.train_mode()}")
        )
        self.ui.pb_train_start.clicked.connect(self.train_start)
        self.ui.pb_train_stop.clicked.connect(self.train_stop)
        self.ui.pb_train_clear.clicked.connect(self.train_clear)
        self.ui.pb_train_export.clicked.connect(self.train_export)
        # 进度条。
        self.ui.progressBar_train.reset()
        self.train_thread = None
        # 指标绘图板块。
        self.init_plot()
        self.is_training = False  # 是否正在训练。
        self.ui.le_update_interval.setValidator(QIntValidator(1, 10, self))
        self.epoch = 0

    def init_config_tab(self):
        "配置面板的初始化。"
        self.config_changed = False

        self.init_filepath_setter(
            self.ui.pb_project_path,
            self.ui.le_project_path,
            isdir=1,
            directory=CODES_DIR,
        )

        self.init_filepath_setter(
            self.ui.pb_data_path,
            self.ui.le_data_path,
            isdir=1,
            directory=self.dataset_dir(),
        )

        ## 架构配置
        self.init_filepath_setter(
            self.ui.pb_backbone_config,
            self.ui.le_backbone_config,
            directory=self.configure_dir(),
        )
        self.init_filepath_setter(
            self.ui.pb_reweight_config,
            self.ui.le_reweight_config,
            directory=self.configure_dir(),
        )
        ## 预训练配置
        self.init_filepath_setter(
            self.ui.pb_data_config_pretrain,
            self.ui.le_data_config_pretrain,
            directory=self.configure_dir(),
        )
        self.init_filepath_setter(
            self.ui.pb_python_path_pretrain,
            self.ui.le_python_path_pretrain,
            directory=python_dir(),
        )
        ## 小样本微调配置
        self.init_filepath_setter(
            self.ui.pb_data_config_fewtune,
            self.ui.le_data_config_fewtune,
            directory=self.configure_dir(),
        )
        self.init_filepath_setter(
            self.ui.pb_model_path_fewtune,
            self.ui.le_model_path_fewtune,
            directory=self.model_dir(),
        )

    def init_plot(self):
        "初始化绘图面板"
        # 白色背景，黑色前景。
        pg.setConfigOption("background", "#FFFFFF")
        pg.setConfigOption("foreground", "k")
        pg.setConfigOption("antialias", True)

        win = pg.GraphicsLayoutWidget()
        layout = self.ui.graph_layout
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(win)

        plt_loss: PlotItem = win.addPlot(title="损失函数")
        # plt_loss = win.addPlot()
        # plt_loss.setLabel("left", text="loss")  # y轴设置函数
        # plt_loss.setLogScale(y=True)
        plt_loss.showGrid(x=True, y=True)  # 栅格设置函数
        # plt_loss.setLabel("bottom", text="epoch")  # x轴设置函数
        plt_loss.addLegend()  # 可选择是否添加legend

        win.nextRow()
        plt_metrics: PlotItem = win.addPlot(title="指标")
        # plt_metrics.setLabel("left", text="metrics")  # y轴设置函数
        # plt_loss.setLogScale(y=True)
        plt_metrics.showGrid(x=True, y=True)  # 栅格设置函数
        # plt_metrics.setLabel("bottom", text="epoch")  # x轴设置函数
        plt_metrics.addLegend()  # 可选择是否添加legend
        # plt_metrics.setMouseEnabled(x=False, y=False)
        self.plt_loss, self.plt_metrics = plt_loss, plt_metrics
        self.clear_plot()

    def clear_plot(self):
        "辅助函数：清空绘图有关的变量"
        plt_loss, plt_metrics = self.plt_loss, self.plt_metrics
        plt_loss.clear()
        plt_metrics.clear()

        self.plt_items: dict[str, PlotDataItem] = {
            StringConstants.loss: plt_loss.plot(name=StringConstants.loss),
            StringConstants.recall: plt_metrics.plot(name=StringConstants.recall),
            StringConstants.precision: plt_metrics.plot(
                name=StringConstants.precision
            ),
        }
        self.plt_pen: dict[str, str] = {
            StringConstants.loss: "b",
            StringConstants.recall: "r",
            StringConstants.precision: "g",
        }

        self.plt_data: dict[str, list] = defaultdict(list)
        self.plt_keys = self.plt_items.keys

    def init_objdet_tab(self):
        """单图识别：初始化"""
        self.input_image_path = None
        self.output_image_path = None
        self.output_image_box = None
        self.thread_objdet = None
        self.ui.progressBar_objdet.reset()
        self.ui.pb_objdet_open.clicked.connect(self.open_objdet)
        self.ui.pb_objdet_detect.clicked.connect(self.detect_objdet)
        self.ui.pb_objdet_export.clicked.connect(self.export_objdet)
        self.ui.pb_objdet_clear.clicked.connect(self.clear_objdet)

    def init_batchdet_tab(self):
        "多图识别：初始化"
        # 多图检测共用单图的线程。
        # self.thread_batchdet = None
        self.ui.progressBar_batchdet.reset()
        self.ui.pb_batchdet_open.clicked.connect(self.open_batchdet)
        self.ui.pb_batchdet_detect.clicked.connect(self.detect_batchdet)
        self.ui.pb_batchdet_export.clicked.connect(self.export_batchdet)
        self.ui.pb_batchdet_clear.clicked.connect(self.clear_batchdet)
        NUM_BATCH = 8
        label_list: list[QLabel] = [
            getattr(self.ui, f"X{i + 1}") for i in range(NUM_BATCH)
        ]
        self.image_panel_list = list(map(ObjdetImagePanel, label_list))
        self.NUM_BATCH = NUM_BATCH

    def total_input_images_batchdet(self):
        "批量识别：当前输入图像的总数"
        total = sum(
            1 for panel in self.image_panel_list if panel.input_image is not None
        )
        return total

    def total_output_images_batchdet(self):
        "批量识别：当前输出图像的总数"
        total = sum(
            1 for panel in self.image_panel_list if panel.output_image is not None
        )
        return total

    def stop_thread_objdet(self):
        t = self.thread_objdet
        if t is None:
            return
        t.quit()
        t.wait()
        t.deleteLater()
        self.thread_objdet = None

    def open_batchdet(self):
        "批量识别：打开输入图像"
        file_list, _ = QFileDialog.getOpenFileNames(
            self,
            f"请选择最多{self.NUM_BATCH}张图片",
            directory=input_image_dir(),
            filter="*.jpg;;*.jpeg",
        )
        if not file_list:
            print("用户没有选择任何文件")
            return
        print(f"用户输入了 {len(file_list)} 个文件")

        # 从第一个没有填充的输入图像开始填充，到满了为止。
        i = 0
        for panel in self.image_panel_list:
            if panel.input_image is not None:
                continue
            if i >= len(file_list):
                break
            panel.input_image = file_list[i]
            panel.output_image = None
            self.show_image(panel.input_image, panel.label)
            print(f"展示图像：{panel.input_image}")
            i += 1

        print(f"新增了{i}张图像，当前图像：{self.total_input_images_batchdet()}")
        if i < len(file_list):
            not_used = len(file_list) - i
            QMessageBox.warning(
                self, "警告", f"空间不足，{not_used} 张图像未被选择。",
            )

    def detect_batchdet(self):
        "批量识别：进行识别"
        if not self.total_input_images_batchdet():
            QMessageBox.warning(self, "错误", "输入图像为空，请打开图像文件进行检测")
            return

        input_image_list = []
        self.input_image_ids = []
        for i, panel in enumerate(self.image_panel_list):
            if panel.input_image is None:  # 用户没有输入
                continue
            if panel.output_image is not None:  # 不要重复检测
                continue
            input_image_list.append(panel.input_image)
            self.input_image_ids.append(i)

        if not input_image_list:
            print(f'所有图像都检测过了，没有需要检测的图像')
            return  # 所有的输入图像都检测过了，没有要检测的。
        self.start_thread_objdet(input_image_list, self.handle_detection_result_batch)

    def handle_detection_result_batch(self, result):
        result_images = result[StringConstants.result]
        images_to_boxes = result[StringConstants.images_to_boxes]
        for i, output_image in zip(self.input_image_ids, result_images):
            panel = self.image_panel_list[i]
            panel.output_image = P(output_image)
            self.run_progress_bar(self.ui.progressBar_batchdet)
            self.show_image(output_image, panel.label)
            print(f"检测面板{i} 输出图像：{output_image}")
            if images_to_boxes is None:
                continue
            panel.box_list = images_to_boxes[panel.output_image.name]
            print(f'检测结果：{panel.box_list}')

    def export_batchdet(self):
        total = self.total_output_images_batchdet()
        if not total:
            # 一张检测好的都没有。
            QMessageBox.warning(self, "错误", "检测结果为空，请先输入图片进行检测")
            return
        output_dir = QFileDialog.getExistingDirectory(self, "选择导出目录", directory=export_dir())
        if not output_dir:
            print(f"用户没有选择目录")
            return
        output_dir = P(output_dir)
        # 中途可能有人把目录删除了。
        if not output_dir.exists():
            QMessageBox.warning(self, "错误", f"导出目录不存在，请重新选择。")
            return

        for i, panel in enumerate(self.image_panel_list):
            if panel.output_image is None:
                continue
            output_image = P(panel.output_image)
            output_file = output_dir.joinpath(output_image.name)
            print(f'导出面板{i} 到 {output_file}')
            self.export_file(
                src_file=output_image,
                dst_file=output_file,
                progress_bar=self.ui.progressBar_batchdet,
            )
            if panel.box_list is None:
                continue  # 不支持锚框导出。
            output_file_box = output_file.with_suffix('.txt')
            print(f'导出锚框文件：{output_file_box}')
            with output_file_box.open('w') as f:
                json.dump(panel.box_list, f, indent=4)

        print(f'导出完成：{total}')

    def clear_batchdet(self):
        print(f"重置所有输入输出图像")
        self.stop_thread_objdet()
        for lb in self.image_panel_list:
            lb.clear()

    def open_objdet(self):
        "单图识别：打开一张图像"
        image_file, _ = QFileDialog.getOpenFileName(
            self, "打开图像", directory=input_image_dir(), filter="*.jpg;;*.jpeg;;"
        )
        if not image_file:
            print("用户没有任何输入")
            return
        if not self.show_image(image_file, self.ui.lb_objdet_input):
            QMessageBox.warning(self, "错误", f"文件{image_file}不是合法的图像格式。")
            return
        print(f"图像展示成功：{image_file}")
        self.input_image_path = image_file
        self.output_image_path = None  # 旧的识别结果作废了。

    def clear_objdet(self):
        "单图检测：清空"
        self.ui.lb_objdet_input.clear()
        self.ui.lb_objdet_output.clear()
        self.input_image_path = None
        self.output_image_path = None
        self.output_image_box = None
        self.stop_thread_objdet()
        print(f"清空完成")

    def export_objdet(self):
        "单图检测：导出检测结果"
        if self.output_image_path is None:
            QMessageBox.warning(self, "错误", "检测结果为空，请先输入一张图片进行检测")
            return
        output_image_path = P(self.output_image_path)
        output_dir = QFileDialog.getExistingDirectory(self, "选择导出目录", directory=export_dir())
        if not output_dir:
            print(f"用户取消了导出")
            return
        # 为了方便，文件名还是原来的文件名。
        output_file = P(output_dir).joinpath(output_image_path.name)
        self.export_file(
            src_file=output_image_path,
            dst_file=output_file,
            progress_bar=self.ui.progressBar_objdet,
        )
        if self.output_image_box is not None:
            output_file_box = output_file.with_suffix('.txt')
            with output_file_box.open('w') as f:
                json.dump(self.output_image_box, f, indent=4)
            print(f'单图检测：导出锚框文件：{output_file_box}')

        print(f'导出完成')

    def export_file(self, src_file: P, dst_file: P, progress_bar: QProgressBar):
        "辅助函数：文件导出"
        src_file, dst_file = map(P, [src_file, dst_file])
        if dst_file.exists():
            print(f"文件已存在")
            rely = QMessageBox.question(
                self,
                "文件已存在",
                f"文件 {dst_file} 已存在，要覆盖吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if rely == QMessageBox.No:
                print("取消了覆盖")
                return
            else:
                print(f"覆盖文件：{dst_file}")

        self.run_progress_bar(progress_bar)
        try:
            # 可能是权限问题，IO异常。
            dst_file.write_bytes(src_file.read_bytes())
        except:
            print(f"文件导出异常：")
            traceback.print_exc()
            QMessageBox.warning(self, "文件导出失败", f"文件导出异常，请查看日志。")
        else:
            print(f"文件导出成功：{dst_file}")

    def detect_objdet(self):
        "单图目标检测"
        # 没有输入图像。
        if self.input_image_path is None:
            QMessageBox.warning(self, "错误", "必须先打开一张输入图像")
            return
        # 已有输出图像。
        if self.output_image_path is not None:
            reply = QMessageBox.question(
                self,
                "提示",
                "输入图像已识别完毕，是否重新检测？",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if reply == QMessageBox.Cancel:
                return
            print(f"用户要求重新检测")
            self.output_image_path = None

        assert os.path.isfile(self.input_image_path) and self.output_image_path is None
        input_image_path = P(self.input_image_path)
        print(f"目标识别中：{input_image_path}")
        self.start_thread_objdet([input_image_path], self.handle_detection_result_single)

    def start_thread_objdet(self, input_image_list, handle_detection_result):
        "启动目标检测线程（start和end信号都有公共代码处理）"
        self.stop_thread_objdet()
        t = ObjdetThread(input_image_list)
        t.detection_error_signal.connect(self.handle_detection_error)
        t.detection_result_signal.connect(handle_detection_result)
        t.start()
        self.thread_objdet = t

    def handle_detection_error(self):
        "处理 error_signal "
        QMessageBox.warning(self, "错误", "目标检测失败，请查看日志输出")
        self.stop_thread_objdet()

    def handle_detection_result_single(self, result):
        result, images_to_boxes = result[StringConstants.result], result[StringConstants.images_to_boxes]
        assert len(result) == 1
        output_image_path = result[0]
        self.run_progress_bar(self.ui.progressBar_objdet)
        self.show_image(output_image_path, self.ui.lb_objdet_output)
        self.output_image_path = output_image_path
        self.stop_thread_objdet()
        print(f"目标检测结果已经展示：{output_image_path}")
        if images_to_boxes is None:
            return
        print(f'单图检测结果：{images_to_boxes}')
        assert len(images_to_boxes) == 1
        self.output_image_box = images_to_boxes

    def show_image(self, image_path, label: QLabel):
        "将一个图片路径显示到label上面"
        image_path = str(image_path)
        image = QtGui.QPixmap(image_path).scaled(label.width(), label.height())
        if image.isNull():
            return False
        label.setPixmap(image)
        return True

    def train_mode(self) -> TrainMode:
        "训练模式的enum值"
        return TEXT_TO_TRAIN_MODE[self.train_mode_raw()]

    def get_config_paths(self) -> dict:
        "辅助函数：返回有关路径的配置"
        config_dict = {}
        config_dict.setdefault("解析器路径", self.python_path())
        config_dict.setdefault("重加权网络配置", self.reweight_config())
        config_dict.setdefault("主干网络配置", self.backbone_config())
        config_dict.setdefault("预训练数据配置", self.pretrain_data_config())
        config_dict.setdefault("小样本微调数据配置", self.fewtune_data_config())
        config_dict.setdefault("小样本微调预训练模型路径", self.fewtune_model_path())
        return config_dict

    def pretrain_command(self):
        command = " ".join(
            [
                self.python_path(),
                "train.py",
                self.pretrain_data_config(),
                self.backbone_config(),
                self.reweight_config(),
            ]
        )
        print(f"预训练命令：{command}")
        return command

    def fewtune_command(self):
        command = " ".join(
            [
                self.python_path(),
                "train.py",
                self.fewtune_data_config(),
                self.backbone_config(),
                self.reweight_config(),
                self.fewtune_model_path(),
            ]
        )
        print(f"小样本微调命令：{command}")
        return command

    def command_for_train_mode(self):
        "根据当前训练模式获取训练命令"
        if self.train_mode() == TrainMode.pretrain:
            cmd = self.pretrain_command()
        else:
            cmd = self.fewtune_command()
        return cmd

    def make_train_log_filename(self):
        current_time = int(time.time())
        mode = self.train_mode().name
        return f'{current_time:015d}-{mode}.txt'

    def train_export(self):
        "训练面板：导出日志"
        if self.is_training:
            QMessageBox.warning(self, "错误", f"正在训练中，无法导出日志")
            return
        train_log = self.console.toPlainText()
        if not train_log:
            print('训练日志为空')
            reply = QMessageBox.question(self, "提示", "训练日志为空，确定要导出？",
                                         QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
            if reply == QMessageBox.Cancel:
                print('取消导出训练日志')
                return

        output_dir = QFileDialog.getExistingDirectory(self, "导出", directory=export_dir())
        if not output_dir:
            print('用户取消了导出')
            return

        filename = self.make_train_log_filename()
        output_file = P(output_dir).joinpath(filename)
        print(f'训练日志导出：{output_file}')
        self.run_progress_bar(self.ui.progressBar_train)
        output_file.write_text(train_log)

    def train_clear(self):
        "训练面板：重置训练状态和UI"
        if self.is_training:
            QMessageBox.warning(self, "错误", f'正在训练中，无法重置，请先停止当前训练')
            return
        print(f'重置训练面板，删除所有数据')
        self.clear_plot()
        self.console.clear()
        self.stop_train_thread()

    def train_start(self):
        "用户：要开始训练，启动训练进程"
        if self.is_training:
            print("已经有训练进程了，无法开始")
            QMessageBox.warning(self, "警告", "已经有一个训练进程正在运行，请等待当前训练完成，或点击终止以停止当前训练。")
            return
        if not self.check_config():
            print("配置非法，训练无法开始")
            return
        # 这里只是拉起训练线程，UI的更新要到线程反馈了才进行。
        # 如果线程没有启动，那么UI就维持不变。
        cmd = self.command_for_train_mode()
        assert self.train_thread is None, "残留的训练线程"
        self.train_thread = self.TrainThreadClass(cmd=cmd, cwd=self.project_dir(), mode=self.train_mode())
        self.train_thread.train_start_signal.connect(self.handle_train_start)
        self.train_thread.train_step_signal.connect(self.handle_train_step)
        self.train_thread.train_end_signal.connect(self.handle_train_end)
        self.train_thread.train_interrupt_signal.connect(self.handle_train_interrupt)
        self.train_thread.start()
        print("训练线程启动")

    def train_stop(self):
        "用户：要终止训练进程"
        if not self.is_training:
            print("当前没有训练进程")
            return
        reply = QMessageBox.question(
            self,
            "确认",
            "当前训练进程还未结束，确认终止当前训练进程吗？",
            QMessageBox.Cancel | QMessageBox.Yes,
            QMessageBox.Cancel,
        )
        if reply == QMessageBox.Cancel:
            print("取消终止训练")
            return
        if self.train_thread is None:
            return
        # 这时候，线程可能发来结束通知，然后就把线程删除了。
        # 但这时又不能把UI冻住，因为训练进程不能暂停。
        # 但这时又不能把UI冻住，因为训练进程不能暂停。
        self.train_thread.requestInterruption()
        # 等待线程发来 interrupt 信号。

    def handle_train_start(self, start_ok: bool):
        "训练线程：启动进程，成功或者失败"
        if not start_ok:
            print("训练线程启动失败")
            # 注意：线程可能已经退出了，但还是需要重置系统状态的。
            self.stop_train_thread()
            QMessageBox.warning(self, "警告", "训练进程启动失败，请检查配置。")
            return

        print("训练线程启动成功，刷新UI")
        self.clear_plot()  # 刷新绘图有关的变量。
        self.console.clear()
        self.console.append(f"当前训练模式：{self.train_mode_raw()}")
        self.console.append("配置文件如下：")
        for name, value in self.get_config_paths().items():
            self.console.append(f"{name}：{value}")
        self.console.append(f"训练命令：{self.train_thread.cmd}")
        self.console.append("正在加载模型和数据集，请等待。")
        self.run_progress_bar(self.ui.progressBar_train)
        self.console.append("加载完成，训练已启动。")
        self.is_training = True
        self.epoch = 0
        # 此时UI已经准备好接收线程的日志。

    def is_mocked(self):
        "训练面板 Mocked"
        return TrainThreadMocked == self.TrainThreadClass

    def handle_train_step(self, line: str):
        "训练线程：发来一行训练日志"
        self.console.append(line)
        self.epoch += 1

        if not is_train_step_line(line):
            return
        if (1 + self.epoch) % self.update_interval() != 0:  # 没到更新周期。
            return
        data = parse_logline(line, METRICS_MAP)
        print(f"解析后的数据：{data}")

        for key in self.plt_keys():
            self.plt_data[key].append(data[key])
            self.plt_items[key].setData(self.plt_data[key], pen=self.plt_pen[key])

    def handle_train_interrupt(self, status):
        "训练线程：已经收到interrupt信号，run返回。"
        print(f"线程被中断，已经退出，善后处理，中断状态码：{status}")
        self.stop_train_thread()
        self.run_progress_bar(self.ui.progressBar_train)
        self.console.append("-" * 10)
        self.console.append("训练进程已终止。")
        print("训练进程已结束")

    def handle_train_end(self, returncode: int):
        "训练线程：训练进程已经退出了。"
        print(f"线程正常退出，进程返回值：{returncode}")
        self.stop_train_thread()
        self.console.append("-" * 10)
        self.console.append("训练结束")
        self.console.append(f"训练进程返回值：{returncode}")
        if returncode:
            QMessageBox.warning(self, "警告", "训练进程异常退出，请检查配置。")

    def stop_train_thread(self):
        "辅助函数：将训练线程完全关闭，释放内存"
        t = self.train_thread
        if t is None:
            assert not self.is_training
            return
        t.quit()
        t.wait()
        t.deleteLater()
        self.train_thread = None
        self.is_training = False

    def run_progress_bar(self, progress_bar, total_steps=5, step_time=0.1):
        progress_bar.reset()
        progress_bar.setRange(0, total_steps)
        for step in range(total_steps):
            # sleep_time = random.gauss(mu=step_time, sigma=step_time / 10)
            time.sleep(step_time)
            progress_bar.setValue(step + 1)
        progress_bar.reset()

    def check_config(self):
        "检查一切路径不为空，不空则肯定是存在的。"
        if not self.config_changed:
            return True
        ok = True

        def check_filepath(name, path):
            nonlocal ok
            if not path:
                QMessageBox.warning(self, "错误", f"{name} 的值不能为空。")
                ok = False
            # 最初输入的路径可能不存在，但如果是Mocked就无所谓了。
            if not MOCKED_TRAIN and not os.path.exists(path):
                QMessageBox.warning(self, "错误", f"{name} 的路径不存在。")
                ok = False

        for name, value in self.get_config_paths().items():
            check_filepath(name, value)

        return ok

    def project_dir(self):
        value = self.ui.le_project_path.text()
        if value and os.path.isdir(value):
            return value
        return None

    def configure_dir(self):
        prj = self.project_dir()
        if prj is None:
            return None
        return os.path.join(prj, "cfg")

    def dataset_dir(self):
        prj = self.project_dir()
        if prj is None:
            return None
        return os.path.join(prj, "dataset")

    def model_dir(self):
        prj = self.project_dir()
        if prj is None:
            return None
        return os.path.join(prj, "backup")

    def init_filepath_setter(
            self, button: QPushButton, lineedit: QLineEdit, isdir=0, directory=None,
    ):
        self.config_changed = True

        def slot():
            if isdir:
                value = QFileDialog.getExistingDirectory(
                    self, "输入目录路径", directory=directory
                )
            else:
                value, _ = QFileDialog.getOpenFileName(
                    self, "输入文件路径", directory=directory
                )

            if not value:
                print(f"用户没有任何输入：{value}")
                return
            lineedit.setText(value)
            print(f"路径设置成功：{value}")

        button.clicked.connect(slot)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MyWindow()
    w.show()
    sys.exit(app.exec_())
