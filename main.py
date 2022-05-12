from Ui_develop import *
import sys
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QPushButton, QLineEdit, QLabel
import os
from pathlib import Path
import shlex
import enum


CODES_DIR = "/home/liao/codes"

# PYTHON_INTERPRETER = "/home/liao/anaconda3/envs/python27/bin/python"

class TrainMode(enum.Enum):
    pretrain = 0
    fewtune = 1

TEXT_TO_TRAIN_MODE = {
    '预训练': TrainMode.pretrain,
    '小样本微调': TrainMode.fewtune,
}

def python_dir():
    return "/home/liao/anaconda3/envs"


class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("小样本遥感图像目标检测")
        self.config_changed = False
        self.is_training = False # 是否正在训练。
        

        # 辅助变量
        self.python_path = self.ui.le_python_path_pretrain.text
        self.reweight_config = self.ui.le_reweight_config.text
        self.backbone_config = self.ui.le_backbone_config.text
        self.pretrain_data_config = self.ui.le_data_config_pretrain.text
        self.fewtune_data_config = self.ui.le_data_config_fewtune.text
        self.fewtune_model_path = self.ui.le_model_path_fewtune.text

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

        # 预训练/小样本微调
        self.ui.pb_pretrain_start.clicked.connect(self.pretrain_command)

    def train_mode(self):
        return TEXT_TO_TRAIN_MODE[self.ui.comboBox_train_mode.currentText()]
        
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
            # reply = QMessageBox.question(self, '重新开始训练', '已经重新开始训练？', QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)

            # if reply == QMessageBox.Yes:
            #     print('退出')
            # else:
            #     print('不退出')

    def pretrain_start(self):
        if self.is_training:
            QMessageBox.warning('已经有一个训练进程，请等待当前训练完成，或点击终止以停止当前训练。')
            return


        self.pretrain_command()

    def fewtune_start(self):
        self.fewtune_command()

    def check_config(self):
        "检查一切路径不为空，不空则肯定是存在的。"
        if not self.config_changed:
            return True
        ok = True

        def check_filepath(name, path):
            nonlocal ok
            if not path:
                QMessageBox.warning(f"{name} 的值不能为空。")
                ok = False

        # self.python_path = self.ui.le_python_path_pretrain.text
        # self.reweight_config = self.ui.le_reweight_config.text
        # self.backbone_config = self.ui.le_backbone_config.text
        # self.pretrain_data_config = self.ui.le_data_config_pretrain.text
        # self.fewtune_data_config = self.ui.le_data_config_fewtune.text
        # self.fewtune_model_path = self.ui.le_model_path_fewtune.text

        check_filepath("解析器路径", self.python_path())
        check_filepath("重加权网络配置", self.reweight_config())
        check_filepath("主干网络配置", self.backbone_config())
        check_filepath("预训练数据配置", self.pretrain_data_config())
        check_filepath("小样本微调数据配置", self.fewtune_data_config())
        check_filepath("小样本微调预训练模型路径", self.fewtune_model_path())

        return ok

    def open_image(self, button: QPushButton, label: QLabel, directory=None):
        def open_show_image():
            value, _ = QFileDialog.getOpenFileName(
                self, "打开图像", directory=directory, filter="*.jpg;;*.jpeg;;"
            )
            if not value:
                print("用户没有任何输入")
                return
            image = QtGui.QPixmap(value).scaled(label.width(), label.height())
            if image.isNull():
                QMessageBox.warning(self, "错误", f"文件{value}不是合法的图像格式。")
                return

            label.setPixmap(image)

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
        dir = self.ui.le_data_path.text()
        if dir and os.path.isdir(dir):
            return os.path.join(dir, "evaluation", "images")
        return None

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
