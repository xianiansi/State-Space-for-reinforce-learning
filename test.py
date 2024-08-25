# import heapq
# import random
# import numpy as np
import sys
#
# # sub_popnum = 3
#     # lamda, _ = uniformpoint(sub_popnum, 2) # 2个目标,3个子种群权重
#     # # 计算余弦相似度
#     # suit_pop = []
#     # for l in lamda:
#     #     pop = []
#     #     for i in range(len(Pop_MOP)):
#     #         pop.append(cosine_similarity(targ_pop_[i],l))
#     #     suit_i = pop.index(min(pop))
#     #     suit_pop.append(Pop_MOP[suit_i])
#     # # 针对每个权重向量上的pareto解进行事件仿真
#     # for event in suit_pop:
#     #     schedule_instance = []
#
# # # 定义迷宫地图
# # maze = [
# #     [0, 0, 0, 0, 0],
# #     [0, 1, 0, 1, 0],
# #     [0, 0, 0, 0, 0],
# #     [0, 1, 1, 1, 0],
# #     [0, 0, 0, 0, 0]
# # ]
# #
# #
# # # 定义启发式函数（估计到达目标的代价）
# # def heuristic(node, goal):
# #     return abs(node[0] - goal[0]) + abs(node[1] - goal[1])
# #
# #
# # # 定义 A* 算法函数
# # def astar(maze, start, goal):
# #     # 定义优先队列
# #     open_list = []
# #     # 定义已经访问过的节点集合
# #     closed_set = set()
# #     # 定义每个节点的父节点，用于记录路径
# #     parent = {}
# #     # 将起始节点加入优先队列
# #     heapq.heappush(open_list, (0, start))
# #
# #     while open_list:
# #         # 弹出优先级最高的节点
# #         current_cost, current_node = heapq.heappop(open_list)
# #
# #         # 如果当前节点是目标节点，返回路径
# #         if current_node == goal:
# #             path = []
# #             while current_node in parent:
# #                 path.append(current_node)
# #                 current_node = parent[current_node]
# #             return path[::-1]  # 返回反向路径
# #
# #
# #         # 将当前节点加入已访问集合
# #         closed_set.add(current_node)
# #
# #         # 遍历当前节点的邻居节点
# #         for neighbor in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
# #             neighbor_node = (current_node[0] + neighbor[0], current_node[1] + neighbor[1])
# #
# #             # 如果邻居节点在地图范围内且不是墙且未访问过
# #             if 0 <= neighbor_node[0] < len(maze) and 0 <= neighbor_node[1] < len(maze[0]) and \
# #                     maze[neighbor_node[0]][neighbor_node[1]] == 0 and neighbor_node not in closed_set:
# #                 # 计算到达邻居节点的代价
# #                 neighbor_cost = current_cost + 1
# #                 # 计算启发式函数的值
# #                 neighbor_heuristic = heuristic(neighbor_node, goal)
# #                 # 计算总代价
# #                 total_cost = neighbor_cost + neighbor_heuristic
# #                 # 将邻居节点加入优先队列
# #                 heapq.heappush(open_list, (total_cost, neighbor_node))
# #                 # 记录邻居节点的父节点
# #                 parent[neighbor_node] = current_node
# #
# #     # 如果没有找到路径，返回空列表
# #     return []
# #
# #
# # # 设置起始点和目标点
# # start = (0, 0)
# # goal = (4, 4)
# #
# # # 运行 A* 算法并输出结果
# # path = astar(maze, start, goal)
# # if path:
# #     print("最短路径为:", path)
# # else:
# #     print("未找到路径")
#
# # open_list=[]
# # heapq.heappush(open_list, (0, 0,2))
# # heapq.heappush(open_list, (-1, 4,0))
# # heapq.heappush(open_list, (3, 4,5))
# # heapq.heappush(open_list, (5, 2,-1))
# # heapq.heappush(open_list, (4, 8,9))
# # heapq.heappush(open_list, (5, 5,0))
# # print(heapq.heappop(open_list))
# # print(heapq.heappop(open_list))
# # print(heapq.heappop(open_list))
#
# # choice_pos_list = np.random.choice(5, 4, replace=False)  # 选4
# # print(choice_pos_list)
#
# nums =[1,2,3,4,5,6,7]
# k = 3
# # len_list= len(nums)
# # k = k % len_list
# # num_new = [0]*len_list
# # num_new[k:] = nums[:len_list-k]
# # num_new[:k] = nums[len_list-k:
# len_list= len(nums)
# k = k % len_list
# nums = nums[::-1]
# nums[:k] = nums[:k][::-1]
# nums[k:] = nums[k:][::-1]
# print(nums)


# import sys
#
# n,x = 5,8
# a_list = [1,2,3,4,10]


# # sorted(a_list,reverse = True)
# a_used = {}
# dp_i = [0] * (len(a_list))
# dp = [dp_i] * (x + 1)
# print(dp)
# print(dp[x+1])
#
# for i in range(0, x+1):
#     if i == a_list[0]:
#         dp[0][i] = 1
#     else:
#         dp[0][i] = -1
# for i in range(1,len(a_list)+1):
#     for j in range(1,x+1):
#         if a_list[i] > j:
#             dp[i][j] = dp[i-1][j]
#         else:
#             dp[i][j] = min(dp[i-1][j-a_list[i]]+1,dp[i-1][j])
#
# print(dp[len(a_list)-1][x])


# def twoSum(nums, target):
#     nums = sorted(nums)
#     print(nums)
#     left = 0
#     right = len(nums) - 1
#     result = []
#     while left < right:
#         if nums[left] + nums[right] == target:
#             result.append(left)
#             result.append(right)
#             return result
#         elif nums[left] + nums[right] < target:
#             left += 1
#         else:
#             right -= 1
#     return []
#
# nums = [3,2,4]
# target = 6
# print(twoSum(nums,target))


# def sort(array1,m,array2,n):
#     new_list = []
#     array1 = sorted(array1)
#     array2 = sorted(array2)
#     i = 0
#     j = 0
#     while i < m and j < n:
#         if array1[i] <= array2[j]:
#             new_list.append(array1[i])
#             i += 1
#         else:
#             new_list.append(array2[j])
#             j += 1
#     # 剩下的放进去
#     if i < m:
#         for k in range(i,m):
#             new_list.append(array1[k])
#     elif j < n:
#         for k in range(j,n):
#             new_list.append(array2[k])
#     return new_list
#
#
# array1,m,array2,n = [1,7,3],3,[6,2],2
# new_list = sort(array1,m,array2,n)
# print(new_list)
#
import sys
# n = sys.stdin.readline().strip()
#
# for line in sys.stdin:
#     a = line.split()

# def func(matrix,M):
#     classified = dict()   # 每个节点的类别
#     DI_class = dict()     # 每个类别的风险
#     co = [5,2]
#     for node in matrix:
#         if node[1] == '*':
#             node_class = node[0]
#             classified[node[0]] = node_class   #开辟一个新的类别
#             if node_class in DI_class:
#                 DI_class[node_class] += co[int(node[2])]*int(node[3])
#             else:
#                 DI_class[node_class] = co[int(node[2])]*int(node[3])
#
#         else:
#             node_class = classified[node[1]]
#             classified[node[0]] = node_class   #该节点的类别等于父节点的类别
#             DI_class[node_class] += co[int(node[2])]*int(node[3])
#     # 计算多少个超出阈值
#     num = 0
#     for key, value in DI_class.items():
#         if value > M:
#             num += 1
#     return num
#
# if __name__ == "__main__":
#
#     n, x = map(int, sys.stdin.readline().strip().split())
#     a_list = [0] + list(map(int, sys.stdin.readline().strip().split()))
#
#     f = [[[float('inf')] * 2 for _ in range(x + 1)] for _ in range(n + 1)]
#     f[0][0][0] = 0
#     for i in range(1, n + 1):
#         for j in range(x + 1):
#             for k in range(2):
#                 f[i][j][k] = min(f[i][j][k], f[i - 1][j][k])
#
#                 if j >= a_list[i] and k > 0:
#                     f[i][j][k] = min(f[i][j][k], f[i - 1][j - a_list[i]][k - 1] + 1)
#                 if j >= a_list[i] // 2:
#                     f[i][j][k] = min(f[i][j][k], f[i - 1][j - a_list[i] // 2][k] + 1)
#     result = min(f[n][x][0], f[n][x][1])
#     if result == float('inf'):
#         print(-1)
#     else:
#         print(result)

import matplotlib.pyplot as plt
import numpy as np

# # 定义三维数组的形状
# num_points = 100  # 数据点数量
# shape = (num_points, 3)  # 形状为 (num_points, 3)，表示有 num_points 行，每行有三列
#
# # 生成随机的三维数据
# A = np.random.rand(*shape)
# B = np.random.rand(*shape)
# C = np.random.rand(*shape)
# D = np.random.rand(*shape)
#
#
#
# # ##对比图
# def CompareFig():
#
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     ax.scatter(A[:, 0], A[:, 1], A[:, 2], marker="^", color="lightcoral", label='Improved MOEA/D')
#     ax.scatter(B[:, 0], B[:, 1], B[:, 2], marker="o", color="purple", label='Original MOEA/D')
#     ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker="d", color="skyblue", label='NSGA-II')
#     ax.scatter(D[:, 0], D[:, 1], D[:, 2], marker="1", color="gold", label='NSGA-III')
#
#     ax.set_xlabel('MakeSpan')
#     ax.set_ylabel('Fatigue')
#     ax.set_zlabel('MachineLoad')  # 给三个坐标轴注明坐标名称
#     plt.title("Comparison Result Diagram")
#     plt.grid(True)
#     plt.legend()
#     plt.show()
# #
# #
# CompareFig()




# # 生成示例数据
# x = np.random.rand(100)
# y = np.random.rand(100)
# z = np.random.rand(100)
# a = np.random.rand(100)
# b = np.random.rand(100)
# c = np.random.rand(100)
#
# # 创建一个 Matplotlib 画布
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 绘制散点图
# ax.scatter(x, y, z,c= 'b')
# ax.scatter(a, b, c,c= 'y')
#
#
# # 将 x 轴反方向放置
# # ax.invert_yaxis()
#
# ax.view_init(elev=20, azim=-45)
#
# # 设置坐标轴标签
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
#
# # 显示图形
# plt.show()


import sys

T = int(sys.stdin.readline().strip())
N_list, M_list = [], []
for i in range(T):
    N_i, M_i = map(int, sys.stdin.readline().strip().split())
    N_list.append(N_i)
    M_list.append(M_i)
for i in range(T):
    N = N_list[i]
    M = M_list[i]
    current_money = N
    Price = 10
    current_price = 10
    ham_num = 0

    while True:
        if M >= 5:
            if current_money >= 5:
                ham_num += 1
                current_money -= 5
                M -= 5
            else:
                break
        else:
            current_price = Price - M
            if current_money >= current_price:
                ham_num += 1
                current_money = current_money - current_price
            else:
                break

    print(ham_num, end=' ')
    print(N - current_money)


import sys

T = int(sys.stdin.readline().strip())
N_list, K_list = [], []
str_list = []
for i in range(T):
    N_i, K_i = map(int, sys.stdin.readline().strip().split())
    str_i = sys.stdin.readline().strip().split()
    N_list.append(N_i)
    K_list.append(K_i)
    str_list.append(str_i)


def cal_val(s):
    sum = 0
    for i in range(len(s) - 1):
        sum += int(s[i:i + 2])
    return sum


def locate(s):
    locate = []
    for i in range(len(s)):
        if s[i] == "1":
            locate.append(i)
    return loacte


def huanyuan(loacte):
    s = ""
    for i in range(N):
        if i in loacte:
            s.join("1")
        else:
            s.join("0")
    return s


def change_list(s):
    loacte = locate(s)
    for i in range(len(loacte)):
        if loacte[i] > 0:
            change_ = loacte
            change_[i] -= 1
            change_list.append(huanyuan(change_))
        if loacte[i] < N - 1:
            change_ = loacte
            change_[i] -= 1
            change_list.append(huanyuan(change_))


for i in range(T):
    N = N_list[i]
    K = K_list[i]
    str_i = str_list[i]
    dp = dict()


    def min(str_i):
        change_s = change_list(s)
        for cha in change_s:
            dp[str_i] = min((dp[str_i], min(cha)))

        return dp[str_i]


    while K:
        min(str_i)
        K -= 1

    print(dp[str_i])
