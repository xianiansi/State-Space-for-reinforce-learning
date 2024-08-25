# 单工序返工

import random
import numpy as np





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



class schedule_instance():
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
        self.nObj = 2  # 目标个数2，完工时间和质量
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


    ## 工序解码
    def decode(self, C):  # 染色体中的一个个体C [OS + MS]
        half_len = len(C) // 2
        tmp = [0 for i in range(self.J_num)]  #解码进度：每个工件加工到第几个工序
        J_seq = C[:half_len]   #工件顺序
        O_seq = np.zeros(half_len, dtype=int)   #对应工序数字
        for i in range(half_len):
            tmp[int(C[i]) - 1] += 1
            O_seq[i] = tmp[int(C[i]) - 1]
        M_seq = C[half_len:]
        return J_seq,O_seq,M_seq #输出工件、工序、设备


    ## 无返工
    def origin_target(self, C):
        J_seq,O_seq,M_seq = self.decode(C)
        half_len = len(C) // 2
        J_temp = np.zeros(self.J_num)  # 每个工序上次的完工时刻
        M_temp = np.zeros(self.M_num)  # 每个机器上次的完工时刻
        Q_cost = 0
        M_busy_period = [[] for _ in range(self.M_type_num[self.O_key - 1])]  #关键设备组每个机器的繁忙时间
        PVal = np.zeros((3, half_len), dtype=int)  # 按染色体顺序记录每道工序加工所需时间、完成时刻、加工机器(0~10)
        sq = dict()      # 原定加工顺序
        sm = dict()      # 原定加工机器
        for i in range(self.M_num):
            sq[i+1] = []


        for i in range(half_len):
            Ji,Oi,Mi = J_seq[i], O_seq[i], M_seq[i]      # 工件、工序、机器（从1开始）
            opera_t = self.OpTime[Ji - 1][Oi - 1]        # 设备的选择不影响工序时间,但与加工位置有关
            M_type = self.OpType[Ji - 1][Oi - 1]         # 设备组
            Mi_total = sum(self.M_type_num[:M_type - 1]) + Mi  # 设备Mi在所有设备中的序号
            t_start = max(J_temp[Ji-1], M_temp[Mi_total-1])
            t_end = t_start + opera_t
            J_temp[Ji - 1], M_temp[Mi_total - 1] = t_end, t_end  # 更新完工时刻
            if M_type == self.O_key:                     # 记录关键设备组的加工时间段
                M_busy_period[Mi - 1].append((t_start, t_end))       # [[(1,2),(6,7)],[(3,4)],[(2,6),(8,9)]]
            PVal[0][i] = opera_t                         # 工件工序加工时长
            PVal[1][i] = t_end                           # 工件工序加工完成时刻
            PVal[2][i] = Mi_total                        # 工件工序加工机器
            sq[Mi_total].append((Ji,Oi))
            sm[(Ji, Oi)] = Mi

            if Oi == 1: # 是第一道工序
                Q_cost += self.RmCost[Ji - 1]
            Q_cost += self.OpCost[M_type - 1][Mi - 1]    # 加工成本与机器有关


        Cmax = max(J_temp)
        target_i = np.array([Cmax,Q_cost])
        PVal_st = [PVal,sq,sm]

        return target_i, PVal_st


    ## 立即返工+右移调度
    def right_shift_target(self, C):
        '''
        :param C: 待解码染色体
        :param PVal: 静态调度每个工件的开始加工时间
        :return: 三目标
        '''
        J_seq, O_seq, M_seq = self.decode(C)
        half_len = len(C) // 2
        Rework_num = np.arange(1, 20)  # 返工次数
        J_temp = np.zeros(self.J_num)  # 每个工序上次的完工时刻
        M_temp = np.zeros(self.M_num)  # 每个机器上次的完工时刻
        Q_cost = 0
        robust = 0

        for i in range(half_len):
            Ji, Oi, Mi = J_seq[i], O_seq[i], M_seq[i]  # 工件、工序、机器（从1开始）
            M_type = self.OpType[Ji - 1][Oi - 1]  # 设备组
            Mi_total = sum(self.M_type_num[:M_type - 1]) + Mi  # 设备Mi在所有设备中的序号

            # 开始加工
            t_start = max(J_temp[Ji - 1], M_temp[Mi_total - 1])
            opera_t = self.OpTime[Ji - 1][Oi - 1]  # 设备的选择不影响工序时间
            if Oi == 1:  # 是第一道工序
                Q_cost += self.RmCost[Ji - 1]
            Q_cost += self.OpCost[M_type - 1][Mi - 1]  # 加工成本与机器有关

            # robust += abs(t_start - origin_st)        # 差距


            # 原地立即开启返工
            if M_type == self.O_key:  # 只有特定类型工序（关键工序返工）设定为工序2
                Rework_prob = exponential_pdf(Rework_num, self.MaType[M_type - 1][Mi - 1])  # 返工概率
                Rework_index = 0
                while np.random.rand() < Rework_prob[Rework_index]:
                    opera_t += self.OpTime[Ji - 1][Oi - 1] * self.MTTR
                    Q_cost += self.OpCost[M_type - 1][Mi - 1]
                    Rework_index += 1

            t_end = t_start + opera_t
            J_temp[Ji - 1], M_temp[Mi_total - 1] = t_end, t_end        # 更新完工时刻

        Cmax = max(J_temp)
        target_i = np.array([Cmax, Q_cost, robust])
        return target_i  #原地返工改动幅度为0


    ## 生成一个种群
    def init_pop(self,pop_size):
        O_num_total = sum(self.O_num)  #C=[1,1,1,1,2,2,3,3,3]->self.O_num=[4,2,3]->O_num_total=9
        J_num = len(self.O_num)
        Chrom = np.zeros((pop_size, O_num_total * 2), dtype=int)
        for i in range(pop_size):  # 第i条染色体
            # 工件序列OS
            OS = np.zeros(O_num_total, dtype=int)
            MS = np.zeros(O_num_total, dtype=int)
            num = 0
            for j in range(J_num):
                for k in range(self.O_num[j]):
                    OS[num] = j + 1  # OS=[ 1  1  2  2  2  3]
                    num += 1
            np.random.shuffle(OS)
            Chrom[i][:O_num_total] = OS
            J_seq,O_seq,_ = self.decode(Chrom[i])
            for n in range(O_num_total):
                M_type = self.OpType[J_seq[n] - 1][O_seq[n] - 1]  # OpType = [[2,2,4],[2,1,3,2],[3,1]]
                MS[n] = random.randint(1, self.M_type_num[M_type - 1])  ##随机选择一个机器
            Chrom[i][O_num_total:] = MS
        return Chrom  # 初始化Chrom为一个种群


    ## 种群的所有目标值
    def target_pop(self, Chroms, scene):
        targs = np.zeros((len(Chroms), self.nObj))
        if scene == 'origin':
            for i in range(len(Chroms)):
                targs[i],_ = self.origin_target(Chroms[i])
        return targs









