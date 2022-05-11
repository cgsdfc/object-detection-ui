from Ui_develop import *
import sys
from PyQt5.QtWidgets import QMessageBox, QFileDialog


# class MyWindow(QtWidgets.QMainWindow, Ui_MainWindow):
class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle('小样本遥感图像目标检测')

        # 配置文件的callback
        ## 基本配置
        self.ui.pb_project_path.clicked.connect(
            self.pb_project_path_fun
        )
        self.ui.pb_data_path.clicked.connect(
            self.pb_data_path_fun
        )
        ## 架构配置
        self.ui.pb_backbone_config.clicked.connect(
            self.pb_backbone_config_fun
        )
        self.ui.pb_reweight_config.clicked.connect(
            self.pb_reweight_config_fun
        )
        ## 预训练配置
        self.ui.pb_data_config_pretrain.clicked.connect(
            self.pb_data_config_pretrain_fun
        )
        self.ui.pb_model_path_pretrain.clicked.connect(
            self.pb_model_path_pretrain_fun
        )
        ## 小样本微调配置
        self.ui.pb_data_config_fewtune.clicked.connect(
            self.pb_data_config_fewtune_fun
        )
        self.ui.pb_model_path_fewtune.clicked.connect(
            self.pb_model_path_fewtune_fun
        )


    def pb_project_path_fun(self):
        pass

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = MyWindow()
    w.show()
    sys.exit(app.exec_())
