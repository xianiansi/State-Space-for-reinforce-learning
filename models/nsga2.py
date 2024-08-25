import random
import numpy as np
from scipy.special import comb  # comb 组合数Cmn
from itertools import combinations
import copy
from algori_func import *
from schedule_instance import schedule_instance


# 初始算法，2目标
class NSGA2():
    # input: pop_size: 种群数量；XOVR：交叉概率；MUTR：变异概率；terminate: 迭代次数；archive：存档数目;T:邻域数量
    # instance: 算例对象
    def __init__(self, pop_size, XOVR, MUTR, terminate, archive, T, instance):
        # 算法输入参数
        self.pop_size = pop_size
        self.XOVR = XOVR
        self.MUTR = MUTR
        self.terminate = int(terminate)
        self.archive = archive  # 存档数目
        self.nObj = 2
        self.terminate = terminate
        self.T = T
        self.instance = instance

    # input： Chroms：进化的种群
    #         scene：决定解码的方式
    #         instance：参数对象
    def evolution(self,Chroms,scene):
        targs = self.instance.target_pop(Chroms, scene)
        Zmin = np.array(np.min(targs, 0)).reshape(1, self.nObj)  # 求理想点（M个目标值 ）[1,1,1]三个目标的群体最小值
        iter = 0
        score = 0

        while iter < self.terminate:
            print("第{name}次迭代".format(name=iter))
            matingpool = random.sample(range(self.pop_size), self.pop_size)  # 就是把1-popsize-1打乱顺序
            N = Chroms.shape[0]
            Chrom1 = Chroms[:int(N / 2)]
            Chrom2 = Chroms[int(N / 2):]
            Chrom_new = copy.deepcopy(Chroms)
            for k in range(N):
                if random.random() < self.XOVR:
                    s1 = Chrom1[np.random.choice(int(N / 2), 1)][0]
                    s2 = Chrom2[np.random.choice(int(N / 2), 1)][0]
                    offspring = self.crossover(s1,s2)
                    Chrom_new[k] = offspring
                if random.random() < self.MUTR:
                    Chrom_new[k] = self.mutation(Chroms[k])

            chrTarg = self.instance.target_pop(Chrom_new, scene)
            Chroms, targs = self.optSelect(Chroms, targs, Chrom_new, chrTarg,scene)
            mixpop = np.concatenate((Chroms, Chrom_new), axis=0)
            mixpopfun = self.instance.target_pop(mixpop,scene)
            iter += 1

        ranks = nonDominationSort(targs)
        # pareto解集的目标
        Pop_MOP = Chroms[ranks == 0]
        targ_pop_ = self.instance.target_pop(Pop_MOP, scene)
        return Pop_MOP, targ_pop_,score


    ############################非支配排序#################################
    def nonDominationSort(self, Chrom,scene):
        targ = self.instance.target_pop(Chrom,scene)
        nPop = Chrom.shape[0]
        nF = targ.shape[1]  # 目标函数的个数
        ranks = np.zeros(nPop, dtype=np.int32)
        nPs = np.zeros(nPop)  # 每个个体p被支配解的个数
        sPs = []  # 每个个体支配的解的集合，把索引放进去
        for i in range(nPop):
            iSet = []  # 解i的支配解集
            for j in range(nPop):
                if i == j:
                    continue
                isDom1 = targ[i] <= targ[j]
                isDom2 = targ[i] < targ[j]
                # 是否支配该解-> i支配j
                if sum(isDom1) == nF and sum(isDom2) >= 1:
                    iSet.append(j)
                    # 是否被支配-> i被j支配
                if sum(~isDom2) == nF and sum(~isDom1) >= 1:
                    nPs[i] += 1
            sPs.append(iSet)  # 添加i支配的解的索引
        r = 0  # 当前等级为 0， 等级越低越好
        indices = np.arange(nPop)
        while sum(nPs == 0) != 0:
            rIdices = indices[nPs == 0]  # 当前被支配数为0的索引
            ranks[rIdices] = r
            for rIdx in rIdices:
                iSet = sPs[rIdx]
                nPs[iSet] -= 1
            nPs[rIdices] = -1  # 当前等级的被支配数设置为负数
            r += 1
        return ranks

    def isDominates(self, s1, s2):  # x是否支配y
        return (s1 <= s2).all() and (s1 < s2).any()

    def optSelect(self, pops, fits, chrPops, chrFits,scene):
        nPop, nChr = pops.shape
        nF = fits.shape[1]
        newPops = np.zeros((nPop, nChr),dtype=int)
        newFits = np.zeros((nPop, nF))
        # 合并父代种群和子代种群构成一个新种群
        MergePops = np.concatenate((pops, chrPops), axis=0)  # 拼接
        MergeFits = np.concatenate((fits, chrFits), axis=0)
        MergeRanks = self.nonDominationSort(MergePops,scene)  # 两个种群合并了，两倍大小的种群
        MergeDistances = self.crowdingDistanceSort(MergePops, MergeRanks)

        indices = np.arange(MergePops.shape[0])
        r = 0
        i = 0
        rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
        while i + len(rIndices) <= nPop:
            newPops[i:i + len(rIndices)] = MergePops[rIndices]  # 精英策略，先把等级小的全都放进新种群中
            newFits[i:i + len(rIndices)] = MergeFits[rIndices]
            r += 1  # 当前等级+1
            i += len(rIndices)
            rIndices = indices[MergeRanks == r]  # 当前等级为r的索引

        if i < nPop:  # 还有一部分没放，就在当前rank按拥挤度才弄个大到小放进去填满
            rDistances = MergeDistances[rIndices]  # 当前等级个体的拥挤度
            rSortedIdx = np.argsort(rDistances)[::-1]  # 按照距离排序 由大到小
            surIndices = rIndices[rSortedIdx[:(nPop - i)]]
            newPops[i:] = MergePops[surIndices]
            newFits[i:] = MergeFits[surIndices]
        return (newPops, newFits)

    ##-----------------------------拥挤度排序算法-----------------------------------
    def crowdingDistanceSort(self, Chrom, ranks):
        fits = np.zeros((Chrom.shape[0],2))
        for i in range(0, self.pop_size):
            fits[i],_ = self.instance.origin_target(Chrom[i])
        nPop = Chrom.shape[0]
        nF = fits.shape[1]  # 目标个数
        dis = np.zeros(nPop)
        nR = ranks.max()  # 最大等级
        indices = np.arange(nPop)
        for r in range(nR + 1):
            rIdices = indices[ranks == r]  # 当前等级种群的索引
            rPops = Chrom[ranks == r]  # 当前等级的种群
            rFits = fits[ranks == r]  # 当前等级种群的适应度
            rSortIdices = np.argsort(rFits, axis=0)  # 对纵向排序的索引
            rSortFits = np.sort(rFits, axis=0)
            fMax = np.max(rFits, axis=0)
            fMin = np.min(rFits, axis=0)
            n = len(rIdices)
            for i in range(nF):
                orIdices = rIdices[rSortIdices[:, i]]  # 当前操作元素的原始位置
                j = 1
                while n > 2 and j < n - 1:
                    if fMax[i] != fMin[i]:
                        dis[orIdices[j]] += (rSortFits[j + 1, i] - rSortFits[j - 1, i]) / \
                                            (fMax[i] - fMin[i])
                    else:
                        dis[orIdices[j]] = np.inf
                    j += 1
                dis[orIdices[0]] = np.inf
                dis[orIdices[n - 1]] = np.inf
        return dis


    def crossover(self,s1, s2):
        # 两点交叉
        half_len = len(s1) // 2
        if random.random() < self.XOVR:
            temp = np.zeros(self.instance.J_num, dtype=int)  # 子代中每个工件已经有几道工序了
            offspring = np.zeros(len(s1), dtype=int)
            at1 = 0  # parent1指针
            at2 = 0  # parent2指针
            at = True  # 从哪个parent复制

            for i in range(half_len):
                while (offspring[i] == 0):  # 直到被赋值
                    if at:  # 从parent1取基因
                        j1 = s1[at1] - 1
                        if temp[j1] < self.instance.O_num[j1]:  # parent1对应的这个基因在子代中还没到达最大工序数
                            offspring[i] = s1[at1]  # 赋值
                            offspring[i + half_len] = s1[at1 + half_len]
                        at1 += 1  # 不管是否赋值，at1指针向后一格
                    else:  # 从parent2取基因
                        j2 = s2[at2] - 1
                        if temp[j2] < self.instance.O_num[j2]:
                            offspring[i] = s2[at2]
                            offspring[i + half_len] = s2[at2 + half_len]
                        at2 += 1
                    at = not at  # 逻辑取反
                temp[offspring[i] - 1] += 1


            # 检查可行性，修正染色体
            J_seq, O_seq, M_seq = self.instance.decode(offspring)
            for n in range(len(J_seq)):
                M_type = self.instance.OpType[J_seq[n] - 1][O_seq[n] - 1]  # OpType = [[2,2,4],[2,1,3,2],[3,1]]
                if M_seq[n] > self.instance.M_type_num[M_type - 1]:  ##随机选择一个机器
                    M_seq[n] = random.randint(1, self.instance.M_type_num[M_type - 1])
        else:
            offspring = random.choice((s1, s2))
        return offspring

    def mutation(self, offspring):
        J_seq, O_seq, M_seq = self.instance.decode(offspring)  # 解码后的
        half_len = len(offspring) // 2
        choice_pos_list = np.random.choice(half_len, 10, replace=False)  # 选20个位置变异
        if random.random() < self.MUTR:
            for pos in choice_pos_list:
                M_type = self.instance.OpType[J_seq[pos] - 1][O_seq[pos] - 1]
                M_seq[pos] = random.randint(1, self.instance.M_type_num[M_type - 1])
        return offspring



    # 画图
    def plot_figure(self, C, PVal):
        # ---------------甘特图-------------
        cst = PVal[1][:] - PVal[0][:]  # 每道工序开始加工时刻
        cpt = PVal[0][:]  # 每道工序的加工时长
        cft = PVal[1][:]  # 每道工序的加工完成时刻
        cmn = PVal[2][:]  # 每道工序的加工机器
        osc = np.tile(C[0:int(len(C)/2)], 1)

        # ---------------甘特图-------------
        plt.figure()
        O_num_total = sum(self.instance.O_num)
        for i in range(O_num_total):
            if cft[i] != 0:
                plt.barh(y=cmn[i], width=cft[i] - cst[i], height=0.8, left=cst[i],
                         color=COLORS[osc[i] % LEN_COLORS], alpha=0.8, edgecolor="black")
                sf = r"$_{%s}$" % cst[i], r"$_{%s}$" % cft[i]
                x = cst[i], cft[i]
                # for j, k in enumerate(sf):  # 时间刻度
                #     plt.text(x=x[j], y=cmn[i], s=k,
                #              rotation="horizontal", va="top", ha="center")
                text = r"${%s}$" % (osc[i])  # 工件编号
                # text1 = r"${w%s}$" % (cwn[i])
                plt.text(x=cst[i] + 0.5 * cpt[i], y=cmn[i], s=text, c="black",
                         rotation="horizontal", va="center", ha="center")
                # plt.text(x=cst[i] + 0.8 * cpt[i], y=cmn[i], s=text1, c="black",
                #          rotation="horizontal", va="center", ha="center")
        plt.ylabel(r"Machine", fontsize=12, fontproperties="Arial")
        plt.xlabel(r"Makespan", fontsize=12, fontproperties="Arial")
        plt.title(r"Gantt Chart", fontsize=14, fontproperties="Arial")
        plt.show()


# 重调度，3目标
class MOEAD_1():
    # input: pop_size: 种群数量；XOVR：交叉概率；MUTR：变异概率；terminate: 迭代次数；archive：存档数目;T:邻域数量
    # instance: 算例对象
    def __init__(self,pop_size,XOVR, MUTR, terminate, archive,T,instance):
        # 算法输入参数
        self.pop_size = pop_size
        self.XOVR = XOVR
        self.MUTR = MUTR
        self.terminate = int(terminate)
        self.archive = archive  # 存档数目
        self.nObj = 3
        self.terminate = terminate
        self.T = T
        self.instance = instance

    # input： Chroms：进化的种群
    #         scene：决定解码的方式
    #         instance：参数对象
    def evolution(self,Chroms,scene):
        lamda, pop_size = uniformpoint(self.pop_size, self.nObj)  # Z是向量,N是向量个数(一般小于POP_SIZE)
        targs = self.instance.target_pop(Chroms, scene)
        Zmin = np.array(np.min(targs, 0)).reshape(1, self.nObj)  # 理想点（M个目标值 ）[1,1,1]三个目标的群体最小值
        B = self.look_neighbor(lamda)

        # 迭代过程
        ranks = nonDominationSort(targs)
        Pop_MOP = Chroms[ranks == 0]
        EPs = copy.deepcopy([list(Pop_MOP[i]) for i in range(len(Pop_MOP))])
        for gen in range(self.terminate):
            print("第{name}次迭代".format(name=gen))
            for i in range(pop_size):
                ## 基因重组，从B(i)中随机选取两个序列k，l
                k = random.randint(0, self.T - 1)
                l = random.randint(0, self.T - 1)
                y = self.crossover(Chroms[B[i][k]], Chroms[B[i][l]])
                y = self.mutation(y)
                t_y, _ = self.instance.origin_target(y)

                ##更新z
                for j in range(len(Zmin[0])):
                    if t_y[j] < Zmin[0][j]:
                        Zmin[0][j] = t_y[j]
                ##更新领域解
                for j in range(len(B[i])):
                    gte_xi = self.Tchebycheff(Chroms[B[i][j]], lamda[B[i][j]], Zmin[0])
                    gte_y = self.Tchebycheff(y, lamda[B[i][j]], Zmin[0])
                    if (gte_y <= gte_xi):
                        Chroms[B[i][j]] = y

                ##更新外部存档
                ep = True  # 决定y是否放进EPs,True就放
                delete = []  # 装EPs个体
                for EP in EPs:  # EPs是所有支配个体，EPs外部存档
                    fun_EP,_ = self.instance.origin_target(EP)
                    if isDominates(fun_EP, t_y) or fun_EP.all() == t_y.all():  # EPs[k]支配y
                        ep = False
                        break
                    elif isDominates(t_y, fun_EP):  # 存在y支配EPs[k]
                        delete.append(EP)  # 准备删除EPs[k]
                if ep:  # 存在y支配EPs[k]或者没有任何支配关系
                    EPs.append(list(y))
                    for delete_i in delete[::-1]:
                        EPs.remove(delete_i)

            if gen % 10 == 0:
                print("%d gen has completed!\n" % gen)

        Zmin = np.array(np.min(targs, 0)).reshape(1, self.nObj)  # 求理想点
        Zmax = np.array(np.max(targs, 0)).reshape(1, self.nObj)  # 求负理想点
        targ_pop = self.instance.target_pop(Chroms,scene)
        c_min = Chroms[np.argmin(Chroms[:, 0])]
        Target_Ep = self.instance.target_pop(EPs,scene)
        score_GD = GD(targ_pop, Target_Ep, Zmin, Zmax)
        score_IGD = IGD(targ_pop, Target_Ep, Zmin, Zmax)
        score_HV = HV(Target_Ep, Zmin, Zmax)
        score = np.array([score_GD, score_IGD, score_HV])
        return c_min, targ_pop, score


    def look_neighbor(self,lamda):
        B = []
        for i in range(len(lamda)):
            temp = []
            for j in range(len(lamda)):
                distance = np.sqrt((lamda[i][0] - lamda[j][0]) ** 2 +
                                   (lamda[i][1] - lamda[j][1]) ** 2)
                temp.append(distance)
            l = np.argsort(temp)
            B.append(l[:self.T])
        return B


    def crossover(self,s1, s2):
        # 两点交叉
        half_len = len(s1) // 2
        if random.random() < self.XOVR:
            temp = np.zeros(self.instance.J_num, dtype=int)  # 子代中每个工件已经有几道工序了
            offspring = np.zeros(len(s1), dtype=int)
            at1 = 0  # parent1指针
            at2 = 0  # parent2指针
            at = True  # 从哪个parent复制

            for i in range(half_len):
                while (offspring[i] == 0):  # 直到被赋值
                    if at:  # 从parent1取基因
                        j1 = s1[at1] - 1
                        if temp[j1] < self.instance.O_num[j1]:  # parent1对应的这个基因在子代中还没到达最大工序数
                            offspring[i] = s1[at1]  # 赋值
                            offspring[i + half_len] = s1[at1 + half_len]
                        at1 += 1  # 不管是否赋值，at1指针向后一格
                    else:  # 从parent2取基因
                        j2 = s2[at2] - 1
                        if temp[j2] < self.instance.O_num[j2]:
                            offspring[i] = s2[at2]
                            offspring[i + half_len] = s2[at2 + half_len]
                        at2 += 1
                    at = not at  # 逻辑取反
                temp[offspring[i] - 1] += 1


            # 检查可行性，修正染色体
            J_seq, O_seq, M_seq = self.instance.decode(offspring)
            for n in range(len(J_seq)):
                M_type = self.instance.OpType[J_seq[n] - 1][O_seq[n] - 1]  # OpType = [[2,2,4],[2,1,3,2],[3,1]]
                if M_seq[n] > self.instance.M_type_num[M_type - 1]:  ##随机选择一个机器
                    M_seq[n] = random.randint(1, self.instance.M_type_num[M_type - 1])
        else:
            offspring = random.choice((s1, s2))
        return offspring

    def mutation(self, offspring):
        J_seq, O_seq, M_seq = self.instance.decode(offspring)  # 解码后的
        half_len = len(offspring) // 2
        choice_pos_list = np.random.choice(half_len, 4, replace=False)  # 选4个位置变异
        if random.random() < self.MUTR:
            for pos in choice_pos_list:
                M_type = self.instance.OpType[J_seq[pos] - 1][O_seq[pos] - 1]
                M_seq[pos] = random.randint(1, self.instance.M_type_num[M_type - 1])
        return offspring

    def Tchebycheff(self, x, lamb, z):
        temp = []
        targ,_= self.instance.origin_target(x)
        for i in range(len(targ)):
            temp.append(np.abs(targ[i] - z[i]) * lamb[i])
        return np.max(temp)

    # 画图
    def plot_figure(self, C, PVal):
        # ---------------甘特图-------------
        cst = PVal[1][:] - PVal[0][:]  # 每道工序开始加工时刻
        cpt = PVal[0][:]  # 每道工序的加工时长
        cft = PVal[1][:]  # 每道工序的加工完成时刻
        cmn = C[len(C)/2:]
        osc = np.tile(C[0:len(C)/2], 1)

        # ---------------甘特图-------------
        plt.figure()
        O_num_total = sum(self.instance.O_num)
        for i in range(O_num_total):
            if cft[i] != 0:
                plt.barh(y=cmn[i], width=cft[i] - cst[i], height=0.8, left=cst[i],
                         color=COLORS[osc[i] % LEN_COLORS], alpha=0.8, edgecolor="black")
                sf = r"$_{%s}$" % cst[i], r"$_{%s}$" % cft[i]
                x = cst[i], cft[i]
                # for j, k in enumerate(sf):  # 时间刻度
                #     plt.text(x=x[j], y=cmn[i], s=k,
                #              rotation="horizontal", va="top", ha="center")
                text = r"${%s}$" % (osc[i])  # 工件编号
                # text1 = r"${w%s}$" % (cwn[i])
                plt.text(x=cst[i] + 0.5 * cpt[i], y=cmn[i], s=text, c="black",
                         rotation="horizontal", va="center", ha="center")
                # plt.text(x=cst[i] + 0.8 * cpt[i], y=cmn[i], s=text1, c="black",
                #          rotation="horizontal", va="center", ha="center")
        plt.ylabel(r"Machine", fontsize=12, fontproperties="Arial")
        plt.xlabel(r"Makespan", fontsize=12, fontproperties="Arial")
        plt.title(r"Gantt Chart", fontsize=14, fontproperties="Arial")
        plt.show()








