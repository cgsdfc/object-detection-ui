# PyQt5 使用例子
import sys
import qdarkstyle
from PyQt5 import QtWidgets
from qdarkstyle.dark.palette import DarkPalette
from qdarkstyle.light.palette import LightPalette

# create the application and the main window
app = QtWidgets.QApplication(sys.argv)
window = QtWidgets.QMainWindow()

# # setup stylesheet
# app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
# or in new API
app.setStyleSheet(qdarkstyle.load_stylesheet(palette=LightPalette(),
qt_api='pyqt5'))

# run
window.show()
app.exec_()