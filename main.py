from Ui_develop import *
import sys


# class MyWindow(QtWidgets.QMainWindow, Ui_MainWindow):
class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle('小样本遥感图像目标检测')
        
        


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = MyWindow()
    w.show()
    sys.exit(app.exec_())
