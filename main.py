from Ui_develop import *
import sys
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QPushButton, QLineEdit, QLabel
from PyQt5.QtCore import QThread, pyqtSignal
import pyqtgraph as pg
from pyqtgraph import PlotDataItem

import os
from pathlib import Path
import shlex
import enum
import time

# import numpy as np
import random
import numpy as np
import subprocess


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

class TrainThreadBase(QThread):
    train_start_signal = pyqtSignal(bool)
    train_step_signal = pyqtSignal(str)
    train_end_signal = pyqtSignal(int)
    train_interrupt_signal = pyqtSignal()


class TrainThreadMocked(TrainThreadBase):

    def __init__(self, cmd: list, cwd: str) -> None:
        super().__init__()
        self.epochs = 20

    def run(self):
        print('训练线程启动')
        self.train_start_signal.emit(True)

        for epoch in range(self.epochs):
            if self.isInterruptionRequested():
                print(f'调用方请求中断')
                self.train_interrupt_signal.emit()
                return
            self.msleep(1000)
            acc = random.random()
            loss = np.exp(-epoch)
            line = f'{epoch} {loss} {acc}'
            print('训练线程发送日志')
            self.train_step_signal.emit(line)
        
        print('训练线程结束')
        rc = int(random.random()>0.5) # 测试异常退出的情况。
        self.train_end_signal.emit(rc)


class TrainThread(TrainThreadBase):

    def __init__(self, cmd: list, cwd: str) -> None:
        super().__init__()
        if isinstance(cmd, str):
            cmd = shlex.split(cmd)
        self.cmd = cmd
        assert os.path.isdir(cwd)
        self.cwd = cwd
        self.p: subprocess.Popen = None # 进程实例

    def handle_interrupt(self):
        "调用方向我发起中断请求。"
        if not self.isInterruptionRequested():
            return False
        if self.p is not None:
            try:
                self.p.terminate()
                self.p.wait()
            except Exception as e:
                print(f'杀死进程时抛出异常：{e}')

        self.train_interrupt_signal.emit()
        return True

    def run(self):
        try:
            p = subprocess.Popen(
                self.cmd,
                shell=False, # shell=True 则cmd可以是str，否则必须是list
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.cwd,
            )
        except Exception as e:
            print(e)
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
            # 清空缓存
            # sys.stdout.flush()
            # sys.stderr.flush()

        # 判断返回码状态
        print(f'训练线程：进程返回值：{p.returncode}')
        self.train_end_signal.emit(p.returncode)


class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.TrainThreadClass = TrainThreadMocked
        # self.TrainThreadClass = TrainThread

        # 辅助变量
        self.python_path = self.ui.le_python_path_pretrain.text
        self.reweight_config = self.ui.le_reweight_config.text
        self.backbone_config = self.ui.le_backbone_config.text
        self.pretrain_data_config = self.ui.le_data_config_pretrain.text
        self.fewtune_data_config = self.ui.le_data_config_fewtune.text
        self.fewtune_model_path = self.ui.le_model_path_fewtune.text
        self.console = self.ui.te_train_logging
        self.train_mode_raw = self.ui.comboBox_train_mode.currentText

        # 配置面板
        self.init_config_tab()
        # 训练面板
        self.init_train_tab()
        # 目标检测
        self.init_objdet_tab()

    def init_train_tab(self):
        "训练面板的初始化"
        # 监控训练模式的变化。
        self.ui.comboBox_train_mode.currentTextChanged.connect(
            lambda: print(f"训练模式改变：{self.train_mode()}")
        )
        # 训练开始按钮
        self.ui.pb_train_start.clicked.connect(self.train_start)
        # 训练终止按钮
        self.ui.pb_train_stop.clicked.connect(self.train_stop)
        # 进度条。
        self.ui.progressBar_train.reset()
        self.train_thread = None
        # 指标绘图板块。
        self.init_train_tab_plot()
        self.is_training = False  # 是否正在训练。

    def init_config_tab(self):
        "配置面板的初始化。"
        self.config_changed = False

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

    def init_train_tab_plot(self):
        "初始化绘图面板"
        # 白色背景，黑色前景。
        pg.setConfigOption("background", "#FFFFFF")
        pg.setConfigOption("foreground", "k")
        pg.setConfigOption('antialias', True)

        win = pg.GraphicsLayoutWidget()
        layout = self.ui.graph_layout
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(win)

        plt_loss = win.addPlot(title="损失函数")
        # plt_loss = win.addPlot()
        # plt_loss.setLabel("left", text="loss")  # y轴设置函数
        # plt_loss.setLogScale(y=True)
        plt_loss.showGrid(x=True, y=True)  # 栅格设置函数
        # plt_loss.setLabel("bottom", text="epoch")  # x轴设置函数
        plt_loss.addLegend()  # 可选择是否添加legend

        win.nextRow()
        plt_metrics = win.addPlot(title="指标")
        # plt_metrics = win.addPlot()
        # plt_metrics.setLabel("left", text="metrics")  # y轴设置函数
        # plt_loss.setLogScale(y=True)
        plt_metrics.showGrid(x=True, y=True)  # 栅格设置函数
        # plt_metrics.setLabel("bottom", text="epoch")  # x轴设置函数
        plt_metrics.addLegend()  # 可选择是否添加legend

        self.plt_loss: PlotDataItem = plt_loss.plot(name='loss')
        self.plt_metrics: PlotDataItem = plt_metrics.plot(name='acc')
        self.loss_list = []
        self.metrics_list = []

    def init_objdet_tab(self):
        self.open_image(
            self.ui.pb_objdet_open, self.ui.lb_objdet_input, directory=self.image_dir(),
        )

    def train_mode(self):
        "训练模式的enum值"
        return TEXT_TO_TRAIN_MODE[self.train_mode_raw()]

    def get_config_dict(self) -> dict:
        "辅助函数：获取当前配置字典"
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
        if self.train_mode == TrainMode.pretrain:
            cmd = self.pretrain_command()
        else:
            cmd = self.fewtune_command()
        return cmd

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
        assert self.train_thread is None, '残留的训练线程'
        self.train_thread = self.TrainThreadClass(cmd=cmd, cwd=self.project_dir())
        self.train_thread.train_start_signal.connect(self.handle_train_start)
        self.train_thread.train_step_signal.connect(self.handle_train_step)
        self.train_thread.train_end_signal.connect(self.handle_train_end)
        self.train_thread.train_interrupt_signal.connect(self.handle_train_interrupt)
        self.train_thread.start()
        print('训练线程启动')

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
            print('训练线程启动失败')
            QMessageBox.warning(self, "警告", "训练进程启动失败，请检查配置。")
            return

        print('训练线程启动成功，刷新UI')
        self.clear_plot() # 刷新绘图有关的变量。
        self.console.clear()
        self.console.append(f"当前训练模式：{self.train_mode_raw()}")
        self.console.append("配置文件如下：")
        for name, value in self.get_config_dict().items():
            self.console.append(f"{name}：{value}")
        self.console.append("正在加载模型和数据集，请等待。")
        self.run_progress_bar(total_steps=5, step_time=0.1)
        self.console.append("加载完成，训练已启动。")
        self.is_training = True
        # 此时UI已经准备好接收线程的日志。

    def is_mocked(self):
        return TrainThreadMocked == self.TrainThreadClass

    def handle_train_step(self, line: str):
        "训练线程：发来一行训练日志"
        print(f'UI 收到一行训练日志：{line}')
        if self.is_mocked():
            epoch, loss, acc = map(float, line.split())
            # 解析日志，更新绘图。
            self.loss_list.append(loss)
            self.metrics_list.append(acc)
            self.plt_loss.setData(self.loss_list, pen='b')
            self.plt_metrics.setData(self.metrics_list, pen='r')
            # 日志要同步打到console
            epoch = int(epoch)
            self.console.append(f'epoch {epoch:04d} loss {loss:.4f} acc {acc*100:.2f}')
        else:
            

    def handle_train_interrupt(self):
        "训练线程：已经收到interrupt信号，run返回。"
        print('线程被中断，已经退出，善后处理')
        self.stop_train_thread()
        self.run_progress_bar(4, 0.1)
        self.console.append("-" * 10)
        self.console.append("训练进程已终止。")
        print("训练进程已结束")

    def handle_train_end(self, returncode: int):
        "训练线程：训练进程已经退出了。"
        print(f'线程正常退出，进程返回值：{returncode}')
        self.stop_train_thread()
        self.console.append("-" * 10)
        self.console.append('训练结束')
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

    def clear_plot(self):
        "辅助函数：清空绘图有关的变量"
        self.plt_loss.clear()
        self.plt_metrics.clear()
        self.loss_list.clear()
        self.metrics_list.clear()

    def run_progress_bar(self, total_steps, step_time):
        self.ui.progressBar_train.reset()
        self.ui.progressBar_train.setRange(0, total_steps)
        for step in range(total_steps):
            sleep_time = random.gauss(mu=step_time, sigma=step_time / 10)
            time.sleep(sleep_time)
            self.ui.progressBar_train.setValue(step + 1)
        self.ui.progressBar_train.reset()

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
