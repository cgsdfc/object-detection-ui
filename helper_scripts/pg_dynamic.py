import pyqtgraph as pg
import numpy as np
import psutil

# 获取CPU使用率的定时回调函数
def get_cpu_info():
    cpu = "%0.2f" % psutil.cpu_percent(interval=1)
    data_list.append(float(cpu))
    print(float(cpu))
    plot.setData(data_list,pen='g')

if __name__ == '__main__':
    data_list = []

    # pyqtgragh初始化
    # 创建窗口
    app = pg.mkQApp()  # 建立app
    win = pg.GraphicsWindow()  # 建立窗口
    win.setWindowTitle(u'pyqtgraph 实时波形显示工具')
    win.resize(800, 500)  # 小窗口大小
    # 创建图表
    historyLength = 100  # 横坐标长度
    p = win.addPlot()  # 把图p加入到窗口中
    p.showGrid(x=True, y=True)  # 把X和Y的表格打开
    p.setRange(xRange=[0, historyLength], yRange=[0, 100], padding=0)
    p.setLabel(axis='left', text='CPU利用率')  # 靠左
    p.setLabel(axis='bottom', text='时间')
    p.setTitle('CPU利用率实时数据')  # 表格的名字
    plot = p.plot()
    
    timer = pg.QtCore.QTimer()
    timer.timeout.connect(get_cpu_info) # 定时刷新数据显示
    timer.start(1000) # 多少ms调用一次
    print(plot)

    app.exec_()
