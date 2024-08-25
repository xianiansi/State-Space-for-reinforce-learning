# import matplotlib.pyplot as plt
# import xlrd
# import pandas as pd
# import numpy as np
# from statsmodels.nonparametric.smoothers_lowess import lowess
#
#
# # 指定 CSV 文件路径
# csv_file = 'result.csv'
# df = pd.read_csv(csv_file)
#
#
# # 获取指定列的数据
# x_values = df['terminate']
# y1_values = df['Cmax']
# y2_values = df['Qcost']
# y3_values = df['Robust']
# y_values = y1_values + y2_values
# y__values = y1_values + y2_values + y3_values
# smoothed_1 = lowess(y1_values, x_values, frac=0.2)   # 使用 Loess 平滑
# smoothed_2 = lowess(y2_values, x_values, frac=0.2)   # 使用 Loess 平滑
# smoothed_3 = lowess(y3_values, x_values, frac=0.2)   # 使用 Loess 平滑
# smoothed_ = lowess(y_values, x_values, frac=0.2)   # 使用 Loess 平滑
# smoothed__ = lowess(y__values, x_values, frac=0.2)   # 使用 Loess 平滑
#
#
# # # 创建图表和子图
# # plt.figure(figsize=(18, 6))  # 设置图表大小
# #
# #
# # # 添加图表标题和标签
# # # 左侧子图
# # plt.subplot(1, 3, 1)  # 创建 1x3 的子图布局，选择第一个子图
# # plt.plot(x_values, y1_values, label='Original_1', color='red', linestyle='dotted', linewidth=2)
# # plt.plot(smoothed_1[:, 0], smoothed_1[:, 1], label='Smoothed_1',color='blue', linestyle='-', linewidth=2)
# # plt.legend()  # 添加图例
# # plt.title('Cmax curve with number of iterations')  # 图表标题
# # plt.xlabel('Iterations')  # X 轴标签
# # plt.ylabel('Cmax')  # Y 轴标签
# #
# # # 右侧子图
# # plt.subplot(1, 3, 2)  # 创建 1x2 的子图布局，选择第2个子图
# # plt.plot(x_values, y2_values, label='Original_2', color='green', linestyle='dotted', linewidth=2)
# # plt.plot(smoothed_2[:, 0], smoothed_2[:, 1], label='Smoothed_2',color='orange', linestyle='-', linewidth=2)
# # plt.legend()  # 添加图例
# # plt.title('Qcost curve with number of iterations')  # 图表标题
# # plt.xlabel('Iterations')  # X 轴标签
# # plt.ylabel('Qcost')  # Y 轴标签
# #
# # # 右侧子图
# # plt.subplot(1, 3, 3)  # 创建 1x3 的子图布局，选择第一个子图
# # plt.plot(x_values, y3_values, label='Original_3', color='orange', linestyle='dotted', linewidth=2)
# # plt.plot(smoothed_3[:, 0], smoothed_3[:, 1], label='Smoothed_3',color='purple', linestyle='-', linewidth=2)
# # plt.legend()  # 添加图例
# # plt.title('Robust curve with number of iterations')  # 图表标题
# # plt.xlabel('Iterations')  # X 轴标签
# # plt.ylabel('Robust')  # Y 轴标签
#
# # 显示图表
# plt.show()
#
# # 创建新图表
# plt.figure(figsize=(18, 6))  # 设置图表大小
#
# plt.subplot(1, 2, 1)
# plt.plot(x_values, y_values, label='Original_objs', color='teal', linestyle='dotted', linewidth=2)
# plt.plot(smoothed_[:, 0], smoothed_[:, 1], label='Smoothed_objs',color='mediumvioletred', linestyle='-', linewidth=2)
# plt.axhline(y=60.6+178.3, color='r', linestyle='-')
# plt.legend()  # 添加图例
# plt.title('weighted 2-Objections curve with number of iterations')  # 图表标题
# plt.xlabel('Iterations')  # X 轴标签
# plt.ylabel('Objections')  # Y 轴标签
#
# plt.subplot(1, 2, 2)
# plt.plot(x_values, y__values, label='Original_objs', color='cornflowerblue', linestyle='dotted', linewidth=2)
# plt.plot(smoothed__[:, 0], smoothed__[:, 1], label='Smoothed_objs',color='tomato', linestyle='-', linewidth=2)
# plt.axhline(y=60.6+178.3, color='r', linestyle='-')
# plt.legend()  # 添加图例
# plt.title('weighted 3-Objections curve with number of iterations')  # 图表标题
# plt.xlabel('Iterations')  # X 轴标签
# plt.ylabel('Objections')  # Y 轴标签
#
#
#
#
# # 显示图表
# plt.show()
#
#
#
#
#
#
#
#
# # import pandas as pd
# # import matplotlib.pyplot as plt
# #
# # # 示例数据
# # data = {'Date': pd.date_range(start='1/1/2022', periods=100),
# #         'Value': pd.Series(range(100)) + pd.Series(5 * pd.np.random.randn(100))}
# #
# # # 创建 DataFrame
# # df = pd.DataFrame(data)
# #
# # # 计算移动平均值（窗口大小为5）
# # df['MA'] = df['Value'].rolling(window=5).mean()
# #
# # # 绘制原始数据和移动平均线
# # plt.plot(df['Date'], df['Value'], label='Original')
# # plt.plot(df['Date'], df['MA'], label='Moving Average (window=5)')
# #
# # # 添加图例和标题
# # plt.legend()
# # plt.title('Line Chart with Moving Average')
# #
# # # 显示图表
# # plt.show()
#
#
#
#
#
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from scipy.signal import savgol_filter
# #
# # # 示例数据
# # x = np.linspace(0, 10, 100)
# # y = np.sin(x) + 0.1 * np.random.randn(100)
# #
# # # 使用 Savitzky-Golay 平滑
# # smoothed = savgol_filter(y, window_length=11, polyorder=2)
# #
# # # 绘制原始数据和平滑曲线
# # plt.plot(x, y, label='Original')
# # plt.plot(x, smoothed, label='Smoothed (Savitzky-Golay)')
# #
# # # 添加图例和标题
# # plt.legend()
# # plt.title('Line Chart with Savitzky-Golay Smoothing')
# #
# # # 显示图表
# # plt.show()
#
#



#  -*-coding:utf8 -*-
from gurobipy import *
from itertools import chain

def rmp(a, c):
    dualArray = []
    try:
        # Create a new model
        m = Model("mip1")   # mip1是模型的名称
        # Create variables
        x = [m.addVar(name='x{name}'.format(name=index)) for index in range(len(a))]  #添加变量名字x0、x1、x2
        # Set objective
        m.setObjective(quicksum(list(chain(
            *[[xi * (ci if indexx == indexc else 0) for indexx, xi in enumerate(x)] for indexc, ci in enumerate(c)]))),
            GRB.MINIMIZE)

        m.addConstr(quicksum(list(chain(
            *[[xi * (ai[0] if indexx == indexa else 0) for indexx, xi in enumerate(x)] for indexa, ai in
              enumerate(a)]))) >= 30, name="c0")
        m.addConstr(quicksum(list(chain(
            *[[xi * (ai[1] if indexx == indexa else 0) for indexx, xi in enumerate(x)] for indexa, ai in
              enumerate(a)]))) >= 20, name="c1")
        m.addConstr(quicksum(list(chain(
            *[[xi * (ai[2] if indexx == indexa else 0) for indexx, xi in enumerate(x)] for indexa, ai in
              enumerate(a)]))) >= 40, name="c2")
        m.update()
        m.optimize()
        #
        con = m.getConstrs()
        for v in m.getVars():
            print('%s = %g' % (v.varName, v.x))
        for i in range(m.getAttr(GRB.Attr.NumConstrs)):
            dualArray.append(con[i].getAttr(GRB.Attr.Pi))  # GRB.Attr.SlackGRB.Attr.Pi
        print('Obj: %g' % m.objVal)
        print('pi:', dualArray)
        r = subp1(dualArray)
        b = subp2(dualArray)
        n = subp3(dualArray)
        maxreduce = max(r[0], b[0], n[0])
        aj = []
        cj = 0
        if maxreduce > 0:
            if maxreduce == r[0]:
                aj = r[1]
                cj = 5
            if maxreduce == b[0]:
                aj = b[1]
                cj = 9
            if maxreduce == n[0]:
                aj = n[1]
                cj = 10
            print(maxreduce, aj)

        else:
            print('切的方式')
            print(a)
            print('每种方式的个数')
            for v in m.getVars():
                print('%s = %g' % (v.varName, v.x))
            print('总价格')
            print('Obj: %g' % m.objVal)
            return a, aj
        a.append(aj)
        c.append(cj)
        rmp(a, c)
    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('master Encountered an attribute error')

def subp1(w):
    dualArray = []
    try:
        # Create a new model
        m = Model("sip1")
        # Create variables
        x1 = m.addVar(lb=0.0, vtype=GRB.INTEGER, name="m1")
        x2 = m.addVar(lb=0.0, vtype=GRB.INTEGER, name="m2")
        x3 = m.addVar(lb=0.0, vtype=GRB.INTEGER, name="m3")
        # Set objective
        m.setObjective(w[0] * x1 + w[1] * x2 + w[2] * x3 - 5, GRB.MAXIMIZE)
        m.addConstr(4 * x1 + 5 * x2 + 7 * x3 <= 9, "c0")
        m.optimize()
        c = m.getConstrs()
        for v in m.getVars():
            print('sub1: %s = %g' % (v.varName, v.x))
            dualArray.append(v.x)
        print('sub1: Obj: %g' % m.objVal)
        return m.objVal, dualArray
    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('sub1 Encountered an attribute error')

def subp2(w):
    dualArray = []
    try:
        # Create a new model
        m = Model("sip2")
        # Create variables
        x1 = m.addVar(lb=0.0, vtype=GRB.INTEGER, name="m1")
        x2 = m.addVar(lb=0.0, vtype=GRB.INTEGER, name="m2")
        x3 = m.addVar(lb=0.0, vtype=GRB.INTEGER, name="m3")
        # Set objective
        m.setObjective(w[0] * x1 + w[1] * x2 + w[2] * x3 - 9, GRB.MAXIMIZE)
        m.addConstr(4 * x1 + 5 * x2 + 7 * x3 <= 14, "c0")
        m.optimize()
        c = m.getConstrs()
        for v in m.getVars():
            print('sub2: %s = %g' % (v.varName, v.x))
            dualArray.append(v.x)
        print('sub2: Obj: %g' % m.objVal)
        return m.objVal, dualArray

    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('sub2 Encountered an attribute error')

def subp3(w):
    dualArray = []
    try:
        # Create a new model
        m = Model("sip3")
        # Create variables
        x1 = m.addVar(lb=0.0, vtype=GRB.INTEGER, name="m1")
        x2 = m.addVar(lb=0.0, vtype=GRB.INTEGER, name="m2")
        x3 = m.addVar(lb=0.0, vtype=GRB.INTEGER, name="m3")
        # Set objective
        m.setObjective(w[0] * x1 + w[1] * x2 + w[2] * x3 - 10, GRB.MAXIMIZE)
        m.addConstr(4 * x1 + 5 * x2 + 7 * x3 <= 16, "c0")
        m.optimize()
        c = m.getConstrs()
        for v in m.getVars():
            print('sub3: %s = %g' % (v.varName, v.x))
            dualArray.append(v.x)
        print('sub3: Obj: %g' % m.objVal)
        return m.objVal, dualArray

    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('sub3 Encountered an attribute error')
if __name__ == '__main__':
    a1 = [2, 0, 0]
    a2 = [1, 1, 0]
    a3 = [0, 0, 1]

    a = []
    a.append(a1)
    a.append(a2)
    a.append(a3)
    c = [5, 5, 5]
    rmp(a, c)