# 单工序返工
import copy
import random
import numpy as np




# rand_num = np.random.rand()
# print(rand_num)


# 定义指数分布的概率密度函数
def exponential_pdf(x, lambd):
    return lambd * np.exp(-lambd * x)

#计算余弦相似度
def cosine_similarity(A, B):
    '''
    param A: A = np.array([1, 2, 3])
    param B: B = np.array([4, 5, 6])
    return: similarity余弦相似度
    '''
    # 计算两个向量的点积
    dot_product = np.dot(A, B)
    # 计算两个向量的模
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    # 计算余弦相似度
    similarity = dot_product / (norm_A * norm_B)
    return similarity


class rework_instance():  # 有返工的案例
    # input: OpTime：工序时间[[6,4,3],[2,8,6,8],[3,7]]共3工件，工件1工序1时间为6
    #        OpType: 工序类型[[2,2,4],[2,1,3,2],[3,1]] 3个工件的工序类型，工件1的工序1属于第二类加工(焊接)，所以在焊接设备中选择
    #        MaType： 机器类型[[0.52,0.43,0.37],[0.51,0.48,0.36],[0.67,0.74,0.36,0.66],[0.77]] 每个设备组的设备质量参数PRO，返工概率，
    #                 第一类加工(铆接)设备组的机器1参数为1.1,指数分布的率参数
    #        RmCost: 原材料成本[16,25,12] 共3工件，工件1的原材料成本为16
    #        OpCost： 生产成本[[1.2,2.2,3.3],[1.1,0.9,1.4],[0.8,1.9,1.2,1.8],[1]] 每个设备组的设备加工成本，第一类加工(铆接)设备组的机器1运行一次成本为1.2
    #        MTTR: 返工占工序时间比例 1.0
    #        O_key: 关键工序，对应设备组 2
    def __init__(self, OpTime, OpType, MaType, RmCost, OpCost, MTTR, O_key):
        # 算例输入参数
        self.OpTime = OpTime
        self.OpType = OpType
        self.MaType = MaType
        self.RmCost = RmCost
        self.OpCost = OpCost
        self.MTTR = MTTR
        self.O_key = O_key
        # 恒定参数
        self.nObj = 3  # 目标个数3，完工时间和质量还有鲁棒性
        self.edgecolor, self.text_color = "black", "black"
        self.rotation, self.va, self.ha = "horizontal", "center", "center"
        self.alpha, self.height = 0.8, 0.8
        # 算例计算参数
        self.J_num = len(self.OpTime)      # 工件数目3
        self.O_num = []
        for i in range(self.J_num):
            self.O_num.append(len(self.OpTime[i]))           # [3,4,2], 3工件，每个工件的工序数
        self.M_type_num = [len(row) for row in self.MaType]  # 每个设备组分别有多少机器[3,3,4,1]
        self.M_num = sum(self.M_type_num)                    # 机器总数10
        self.M_key = self.MaType[self.O_key - 1]             # [0.51,0.48,0.36], 关键工序机器组
        # self.alg = alg


    ## 工序解码,包括解码进度
    def decode(self, C, pro_progress, re_progress):
        '''
        :param C: numpy染色体数组，染色体中的一个个体双层的C [工件、设备],返工次数为0表示第一次加工
        :param pro_progress: 状态变量：当前工序，使用dict字典，每个工件的加工进度，前面已经加工了多少个工序
        :param re_progress: 状态变量：当前返工次数
        :return: #输出工件、工序、返工次数、设备
        '''
        half_len = len(C) // 2
        J_seq = C[:half_len]                    # 工件
        O_seq = np.zeros(half_len, dtype=int)   # 工序
        re_seq = np.zeros(half_len, dtype=int)  # 返工
        tmp = dict()                             # 每个工件在这个染色体中第几次出现
        for i in range(half_len):
            if C[i] in tmp:                      # 如果是在该染色体中不是第一次出现
                tmp[C[i]] += 1                   # 第几次出现
                re_seq[i] = 0                    # 未返工
            else:
                tmp[C[i]] = 1                    # 第一次出现
                re_seq[i] = re_progress[C[i]]    # 该工序返工的次数
            O_seq[i] = pro_progress[int(C[i])] + tmp[int(C[i])]  # 目前属于第几道工序
        M_seq = C[half_len:]
        return J_seq, O_seq, re_seq, M_seq  # 输出工件、工序、返工次数、设备


    ## 按照当前的进度(输入一个原地返工的染色体作为样本)，生成一个种群
    def init_pop_3(self, C, pro_progress, re_progress, popsize):
        '''
        :param C: 染色体
        :param pro_progress: 状态变量：当前工序
        :param re_progress: 状态变量：当前返工次数
        :return: 符合当前状态变量的种群
        '''
        O_num_total = len(C)//2  # C包含工件和机器，染色体有多少个工序要安排
        Chrom = np.zeros((popsize, O_num_total * 2), dtype=int)
        for i in range(popsize):  # 第i条染色体
            # 工件序列OS
            OS = C[:O_num_total]
            MS = np.zeros(O_num_total, dtype=int)
            np.random.shuffle(OS)
            Chrom[i][:O_num_total] = OS
            J_seq, O_seq, re_seq, M_seq = self.decode(Chrom[i], pro_progress, re_progress)
            for n in range(O_num_total):
                M_type = self.OpType[J_seq[n] - 1][O_seq[n] - 1]  # OpType = [[2,2,4],[2,1,3,2],[3,1]]
                MS[n] = random.randint(1, self.M_type_num[M_type - 1])  ##随机选择一个机器
            Chrom[i][O_num_total:] = MS
        return Chrom  # 初始化Chrom为一个种群


    ## 完全重调度
    def complete_re_target(self, C, sq, sm, alg):
        '''
        :param C: 静态调度得到的染色体
        :param sq: {M1:[O11,O21],M2:[O22,O12]}静态调度中每个设备上的加工顺序
        :param sm: {(1,1):4,(1,2):2,(2,1):1,(2,2):3}每个工序对应的加工机器
        :param alg: 使用的算法
        :return: 最终考虑返工并实时重调度的三个目标值
        '''
        # 初始化状态变量、每个工序的完工时刻
        half_len = len(C) // 2
        pro_progress = dict()   # 工序进度
        re_progress = dict()    # 返工进度
        for i in range(half_len):
            pro_progress[C[i]] = 0     # 每个工件的工序进度为0
            re_progress[C[i]] = 0      # 上一个工序的返工进度为0
        J_seq, O_seq, re_seq, M_seq = self.decode(C, pro_progress, re_progress)
        Rework_num = np.arange(1, 20)  # 返工次数
        J_temp = np.zeros(self.J_num)  # 每个工序上次的完工时刻
        M_temp = np.zeros(self.M_num)  # 每个机器上次的完工时刻
        sq_new = dict()                # 每个机器的加工序列
        for i in range(1, self.M_num+1):
            sq_new[i] = []
        Q_cost = 0
        robust = 0


        while len(J_seq):  # 还有工件没做完
            Ji, Oi, Ri, Mi = J_seq[0], O_seq[0], re_seq[0], M_seq[0]  # 工件、工序、返工、机器（从1开始）
            M_type = self.OpType[Ji - 1][Oi - 1]               # 设备组/加工类别
            Mi_total = sum(self.M_type_num[:M_type - 1]) + Mi  # 设备Mi在所有设备中的序号


            ## 开始加工（可能是返工工序也可能是第一次加工）
            t_start = max(J_temp[Ji - 1], M_temp[Mi_total - 1])
            if Ri:
                opera_t = self.OpTime[Ji - 1][Oi - 1] * self.MTTR      # 如果是返工工序，则时间乘以MTTR
            else:
                opera_t = self.OpTime[Ji - 1][Oi - 1]                  # 如果是首次加工工序，则时间不变
            t_end = t_start + opera_t
            if Oi == 1:  # 是第一道工序
                Q_cost += self.RmCost[Ji - 1]
            Q_cost += self.OpCost[M_type - 1][Mi - 1]  # 加工成本与机器有关
            origin_m = sm[(Ji, Oi)]             # 原定设备
            if origin_m != Mi:                  # 设备改动
                robust += 0.5                   # 顺序变化
            if (Ji,Oi) in sq[Mi_total]:
                ind = sq[Mi_total].index((Ji,Oi))
                if ind > 1 and sq_new[Mi_total] and sq_new[Mi_total][-1] != sq[Mi_total][ind - 1]:
                    robust += 0.5

            # 修改存储时间
            J_temp[Ji - 1], M_temp[Mi_total - 1] = t_end, t_end
            # 修改机器加工序列
            sq_new[Mi_total].append((Ji, Oi))
            # 修改目前的染色体
            half_len = len(C) // 2
            C = np.delete(C, [0,half_len])              # 删掉工序和机器


            # 返工工序加入到序列中
            if M_type == self.O_key:         # 只有特定类型工序（关键工序返工）设定为工序2
                Rework_prob = exponential_pdf(Rework_num, self.MaType[M_type - 1][Mi - 1])  # 返工概率
                if np.random.rand() < Rework_prob[Ri]:  # 后面需要返工
                    # 修改目前的染色体
                    C = np.insert(C, 0, Ji)         # 在0索引处增加工序
                    C = np.insert(C, half_len, Mi)  # 增加机器Mi
                    re_progress[Ji] += 1
                    Chrom = self.init_pop_3(C, pro_progress, re_progress, alg.pop_size)

                    # 动态多目标优化
                    Pop_MOP, targ_pop_, score = alg.evolution(Chrom, J_temp, M_temp, pro_progress, re_progress, sq, sm)
                    pop = []
                    l = np.array([1, 1, 1])
                    for i in range(len(Pop_MOP)):
                        pop.append(cosine_similarity(targ_pop_[i], l))
                    suit_i = pop.index(min(pop))
                    C = Pop_MOP[suit_i]
                else:   # 不需要返工
                    re_progress[Ji] = 0
                    pro_progress[Ji] += 1
            else:       # 普通工序（不考虑返工不返工）
                pro_progress[Ji] += 1

            if len(C):
                J_seq, O_seq, re_seq, M_seq = self.decode(C, pro_progress, re_progress)
            else:
                J_seq = []

        Cmax = max(J_temp)
        target_i = np.array([Cmax, Q_cost, robust])
        return target_i


    # 完全重调度（画图版）
    def complete_re_target_plot(self, C, sq, sm, alg,ax):
        '''
        :param C: 静态调度得到的染色体
        :param sq: {M1:[O11,O21],M2:[O22,O12]}静态调度中每个设备上的加工顺序
        :param sm: {(1,1):4,(1,2):2,(2,1):1,(2,2):3}每个工序对应的加工机器
        :param alg: 使用的算法
        :return: 最终考虑返工并实时重调度的三个目标值
        '''
        # 初始化状态变量、每个工序的完工时刻
        half_len = len(C) // 2
        pro_progress = dict()   # 工序进度
        re_progress = dict()    # 返工进度
        for i in range(half_len):
            pro_progress[C[i]] = 0     # 每个工件的工序进度为0
            re_progress[C[i]] = 0      # 上一个工序的返工进度为0
        J_seq, O_seq, re_seq, M_seq = self.decode(C,pro_progress,re_progress)
        Rework_num = np.arange(1, 20)  # 返工次数
        J_temp = np.zeros(self.J_num)  # 每个工序上次的完工时刻
        M_temp = np.zeros(self.M_num)  # 每个机器上次的完工时刻
        sq_new = dict()                # 每个机器的加工序列
        for i in range(1,self.M_num+1):
            sq_new[i] = []
        Q_cost = 0
        robust = 0


        while len(J_seq):  # 还有工件没做完
            Ji, Oi, Ri, Mi = J_seq[0], O_seq[0], re_seq[0], M_seq[0]  # 工件、工序、返工、机器（从1开始）
            M_type = self.OpType[Ji - 1][Oi - 1]               # 设备组/加工类别
            Mi_total = sum(self.M_type_num[:M_type - 1]) + Mi  # 设备Mi在所有设备中的序号


            ## 开始加工（可能是返工工序也可能是第一次加工）
            t_start = max(J_temp[Ji - 1], M_temp[Mi_total - 1])
            if Ri:
                opera_t = self.OpTime[Ji - 1][Oi - 1] * self.MTTR      # 如果是返工工序，则时间乘以MTTR
            else:
                opera_t = self.OpTime[Ji - 1][Oi - 1]                  # 如果是首次加工工序，则时间不变
            t_end = t_start + opera_t
            if Oi == 1:  # 是第一道工序
                Q_cost += self.RmCost[Ji - 1]
            Q_cost += self.OpCost[M_type - 1][Mi - 1]  # 加工成本与机器有关
            origin_m = sm[(Ji, Oi)]             # 原定设备
            if origin_m != Mi:                  # 设备改动
                robust += 0.5                   # 顺序变化
            if (Ji,Oi) in sq[Mi_total]:
                ind = sq[Mi_total].index((Ji,Oi))
                if ind > 1 and sq_new[Mi_total] and sq_new[Mi_total][-1] != sq[Mi_total][ind - 1]:
                    robust += 0.5

            # 修改存储时间
            J_temp[Ji - 1], M_temp[Mi_total - 1] = t_end, t_end
            # 修改机器加工序列
            sq_new[Mi_total].append((Ji, Oi))
            # 修改目前的染色体
            half_len = len(C) // 2
            C = np.delete(C, [0, half_len])              # 删掉工序和机器


            # 返工工序加入到序列中
            if M_type == self.O_key:         # 只有特定类型工序（关键工序返工）设定为工序2
                Rework_prob = exponential_pdf(Rework_num, self.MaType[M_type - 1][Mi - 1])  # 返工概率
                if np.random.rand() < Rework_prob[Ri]:  # 后面需要返工
                    # 修改目前的染色体
                    C = np.insert(C, 0, Ji)         # 在0索引处增加工序
                    C = np.insert(C, half_len, Mi)  # 增加机器Mi
                    re_progress[Ji] += 1
                    Chrom = self.init_pop_3(C, pro_progress, re_progress, alg.pop_size)

                    # 动态多目标优化
                    Pop_MOP, targ_pop_, score = alg.evolution(Chrom, J_temp, M_temp, pro_progress, re_progress, sq, sm)  # 静态调度
                    pop = []
                    # 选时间最小的suitable
                    # l = np.array([1, 1, 1])
                    # for i in range(len(Pop_MOP)):
                        # pop.append(cosine_similarity(targ_pop_[i], l))
                    suit_i = pop.index(min(pop[:,0]))
                    C = Pop_MOP[suit_i]
                    # # 每一次返工生成的种群
                    # if len(Chrom) > 1:
                    #     targs = self.target_pops(Chrom, J_temp, M_temp, pro_progress, re_progress, sq, sm)
                    #     ScatterPlot_3D(ax, targs, alg.terminate)
                else:   # 不需要返工
                    re_progress[Ji] = 0
                    pro_progress[Ji] += 1
            else:       # 普通工序（不考虑返工不返工）
                pro_progress[Ji] += 1

            if len(C):
                J_seq, O_seq, re_seq, M_seq = self.decode(C, pro_progress, re_progress)
            else:
                J_seq = []

        Cmax = max(J_temp)
        target_i = np.array([Cmax, Q_cost, robust])
        return target_i

    def target_pops(self, Chrom, J_temp, M_temp, pro_progress, re_progress, sq, sm):
        targs = np.zeros((len(Chrom), self.nObj))
        for i in range(len(Chrom)):
            C = Chrom[i]
            targs[i] = self.target_pop(C, J_temp, M_temp, pro_progress, re_progress, sq, sm)
        return targs

    def target_pop(self, C, J_temp, M_temp, pro_progress, re_progress, sq, sm):
        J_new = copy.deepcopy(J_temp)
        M_new = copy.deepcopy(M_temp)
        J_seq, O_seq, re_seq, M_seq = self.decode(C, pro_progress, re_progress)
        half_len = len(C) // 2
        Q_cost_i = 0
        robust = 0
        sq_new = dict()  # 原定加工顺序
        for i in range(self.M_num):
            sq_new[i + 1] = []

        for i in range(half_len):
            Ji, Oi, Ri, Mi = J_seq[i], O_seq[i], re_seq[i], M_seq[i]  # 工件、工序、返工、机器（从1开始）
            if Ri:
                opera_t = self.OpTime[Ji - 1][Oi - 1] * self.MTTR
            else:
                opera_t = self.OpTime[Ji - 1][Oi - 1]
            M_type = self.OpType[Ji - 1][Oi - 1]  # 设备组
            Mi_total = sum(self.M_type_num[:M_type - 1]) + Mi  # 设备Mi在所有设备中的序号
            t_start = max(J_new[Ji - 1], M_new[Mi_total - 1])
            t_end = t_start + opera_t
            J_new[Ji - 1], M_new[Mi_total - 1] = t_end, t_end  # 更新完工时刻
            origin_m = sm[(Ji, Oi)]  # 原定设备
            if origin_m != Mi:  # 设备改动
                robust += 0.5  # 顺序变化
            if (Ji, Oi) in sq[Mi_total]:
                ind = sq[Mi_total].index((Ji, Oi))
                if ind > 1 and sq_new[Mi_total] and sq_new[Mi_total][-1] != sq[Mi_total][ind - 1]:
                    robust += 0.5
            sq_new[Mi_total].append((Ji, Oi))
            if Oi == 1:  # 是第一道工序
                Q_cost_i += self.RmCost[Ji - 1]
            Q_cost_i += self.OpCost[M_type - 1][Mi - 1]  # 加工成本与机器有关
        Cmax = max(J_new)

        return np.array([Cmax, Q_cost_i, robust])










