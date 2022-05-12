from Ui_develop import *
import sys
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QPushButton, QLineEdit, QLabel
from PyQt5.QtCore import QThread, pyqtSignal
import pyqtgraph as pg

import os
from pathlib import Path
import shlex
import enum
import time

# import numpy as np
import random
import numpy as np


CODES_DIR = "/home/liao/codes"

# PYTHON_INTERPRETER = "/home/liao/anaconda3/envs/python27/bin/python"


class TrainMode(enum.Enum):
    pretrain = 0
    fewtune = 1


TEXT_TO_TRAIN_MODE = {
    "预训练": TrainMode.pretrain,
    "小样本微调": TrainMode.fewtune,
}


def python_dir():
    return "/home/liao/anaconda3/envs"


class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))


class TrainThread(QThread):
    train_start_signal = pyqtSignal(name="train_start")
    train_step_signal = pyqtSignal(dict, name="train_step")
    train_stop_signal = pyqtSignal(int, name="train_stop")

    def run(self):
        pass


class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.config_changed = False
        self.is_training = False  # 是否正在训练。

        # 辅助变量
        self.python_path = self.ui.le_python_path_pretrain.text
        self.reweight_config = self.ui.le_reweight_config.text
        self.backbone_config = self.ui.le_backbone_config.text
        self.pretrain_data_config = self.ui.le_data_config_pretrain.text
        self.fewtune_data_config = self.ui.le_data_config_fewtune.text
        self.fewtune_model_path = self.ui.le_model_path_fewtune.text
        self.console = self.ui.te_train_logging
        self.train_mode_raw = self.ui.comboBox_train_mode.currentText

        # 配置文件的callback

        ## 基本配置
        self.filepath_setter(
            self.ui.pb_project_path,
            self.ui.le_project_path,
            isdir=1,
            directory=CODES_DIR,
        )

        self.filepath_setter(
            self.ui.pb_data_path,
            self.ui.le_data_path,
            isdir=1,
            directory=self.dataset_dir(),
        )

        ## 架构配置
        self.filepath_setter(
            self.ui.pb_backbone_config,
            self.ui.le_backbone_config,
            directory=self.configure_dir(),
        )
        self.filepath_setter(
            self.ui.pb_reweight_config,
            self.ui.le_reweight_config,
            directory=self.configure_dir(),
        )
        ## 预训练配置
        self.filepath_setter(
            self.ui.pb_data_config_pretrain,
            self.ui.le_data_config_pretrain,
            directory=self.configure_dir(),
        )
        self.filepath_setter(
            self.ui.pb_python_path_pretrain,
            self.ui.le_python_path_pretrain,
            directory=python_dir(),
        )
        ## 小样本微调配置
        self.filepath_setter(
            self.ui.pb_data_config_fewtune,
            self.ui.le_data_config_fewtune,
            directory=self.configure_dir(),
        )
        self.filepath_setter(
            self.ui.pb_model_path_fewtune,
            self.ui.le_model_path_fewtune,
            directory=self.model_dir(),
        )

        # 目标检测
        self.open_image(
            self.ui.pb_objdet_open, self.ui.lb_objdet_input, directory=self.image_dir(),
        )

        # 训练模式
        self.ui.comboBox_train_mode.currentTextChanged.connect(
            lambda: print(f"训练模式改变：{self.train_mode()}")
        )
        self.ui.pb_train_start.clicked.connect(self.train_start)
        self.ui.pb_train_stop.clicked.connect(self.train_stop)
        self.ui.progressBar_train.reset()
        self.init_plot()

    def update_plot(self):
        t = np.linspace(0, 20, 200)
        loss = np.exp(-t)
        metrics = np.exp(t)
        self.plt_loss.plot(t, loss, pen="b", name="loss")
        self.plt_metrics.plot(t, metrics, pen="r", name="ACC")

    def init_plot(self):
        # pg.setConfigOption(antialias=True)
        pg.setConfigOption("background", "#FFFFFF")
        pg.setConfigOption("foreground", "k")

        win = pg.GraphicsLayoutWidget()
        layout = self.ui.graph_layout
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(win)

        # plt_loss = win.addPlot(title="损失函数")
        plt_loss = win.addPlot()
        plt_loss.setLabel("left", text="loss")  # y轴设置函数
        # plt_loss.setLogScale(y=True)
        plt_loss.showGrid(x=True, y=True)  # 栅格设置函数
        # plt_loss.setLabel("bottom", text="epoch")  # x轴设置函数
        plt_loss.addLegend()  # 可选择是否添加legend

        win.nextRow()
        # plt_metrics = win.addPlot(title="指标")
        plt_metrics = win.addPlot()
        plt_metrics.setLabel("left", text="metrics")  # y轴设置函数
        # plt_loss.setLogScale(y=True)
        plt_metrics.showGrid(x=True, y=True)  # 栅格设置函数
        # plt_metrics.setLabel("bottom", text="epoch")  # x轴设置函数
        plt_metrics.addLegend()  # 可选择是否添加legend

        self.plt_loss = plt_loss
        self.plt_metrics = plt_metrics

    def train_mode(self):
        return TEXT_TO_TRAIN_MODE[self.train_mode_raw()]

    def get_config_dict(self) -> dict:
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

    def train_start(self):
        if self.is_training:
            print("已经有训练进程了，无法开始")
            QMessageBox.warning(self, "警告", "已经有一个训练进程正在运行，请等待当前训练完成，或点击终止以停止当前训练。")
            return
        if not self.check_config():
            print("配置非法，训练无法开始")
            return
        self.console.clear()
        self.console.append(f"当前训练模式：{self.train_mode_raw()}")
        self.console.append("配置文件如下：")
        for name, value in self.get_config_dict().items():
            self.console.append(f"{name}：{value}")
        self.console.append("正在加载模型和数据集，请等待。")
        # TODO: 加载训练模型
        self.run_progress_bar(total_steps=5, step_time=0.1)
        self.console.append("加载完成，训练已启动。")
        self.is_training = True
        self.clear_plot()
        self.update_plot()

    def clear_plot(self):
        self.plt_loss.clear()
        self.plt_metrics.clear()

    def run_progress_bar(self, total_steps, step_time):
        self.ui.progressBar_train.reset()
        self.ui.progressBar_train.setRange(0, total_steps)
        for step in range(total_steps):
            sleep_time = random.gauss(mu=step_time, sigma=step_time / 10)
            time.sleep(sleep_time)
            self.ui.progressBar_train.setValue(step + 1)
        self.ui.progressBar_train.reset()

    def train_stop(self):
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
        # TODO：等待模型停止。
        self.console.append("正在终止训练进程，请等待。")
        # 进度条。
        self.run_progress_bar(4, 0.1)
        self.console.append("训练进程已停止。")
        self.is_training = False
        print("训练进程已结束")

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

        for name, value in self.get_config_dict().items():
            check_filepath(name, value)

        return ok

    def put_image(self, image_path, label: QLabel):
        "将一个图片路径显示到label上面"
        image = QtGui.QPixmap(image_path).scaled(label.width(), label.height())
        if image.isNull():
            return False
        label.setPixmap(image)
        return True

    def open_image(self, button: QPushButton, label: QLabel, directory=None):
        def open_show_image():
            value, _ = QFileDialog.getOpenFileName(
                self, "打开图像", directory=directory, filter="*.jpg;;*.jpeg;;"
            )
            if not value:
                print("用户没有任何输入")
                return
            if not self.put_image(value, label):
                QMessageBox.warning(self, "错误", f"文件{value}不是合法的图像格式。")
                return

        button.clicked.connect(open_show_image)

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

    def image_dir(self):
        return "/home/liao/codes/Object_Detection_UI/images/input"

    def filepath_setter(
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
