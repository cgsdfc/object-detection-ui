from Ui_develop import *
import sys
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QPushButton, QLineEdit, QLabel
import os
from pathlib import Path

CODES_DIR = "/home/liao/codes"


class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("小样本遥感图像目标检测")

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
            self.ui.pb_model_path_pretrain,
            self.ui.le_model_path_pretrain,
            directory=self.model_dir(),
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
