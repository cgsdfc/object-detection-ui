from Ui_develop import *
import sys


class MyWindow(QtWidgets.QMainWindow, Ui_MainWindowv):
    def __init__(self):
        super(MyWindow, self).__init__()

        self.setupUi(self)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = MyWindow()
    w.show()
    sys.exit(app.exec_())
