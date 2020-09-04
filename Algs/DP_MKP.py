# coding: utf-8
# 背包问题

import numpy as np


# 1). 0-1 背包问题


def DP_bag01_dim2(N, Weight, Value, K):
    # Init
    # M = [[-1 for _ in range(K+1)] for _ in range(N+1)]
    M = np.full([N + 1, K + 1], -1)
    for w in range(0, K + 1):
        M[0, w] = 0
    for i in range(1, N + 1):
        M[i, 0] = 0
    # Compute
    for i in range(1, N + 1):
        for C in range(1, K + 1):
            if Weight[i-1] > C:
                M[i, C] = M[i -1, C]
            else:
                M[i, C] = max(M[i -1, C], M[i -1, C - Weight[i-1]] + Value[i-1])
    # Output
    print(M)
    return M[N, C]

def DP_bag01_dim1roll(N, Weight, Value, K):
    # Init: maxValue
    Weight = [0] + Weight
    Value = [0] + Value
    # S = [[-1 for _ in range(2 ** N)] for _ in range(2)]
    p = 0  # 作用是指向数组的某一行 (两行其中之一), 不能再用下标 i 来指定数组的行数了
    # maxValue = [[0 for _ in range(2 ** N + 1)] for _ in range(2)]
    maxValue = np.full([2, K + 1], 0)
    # Calc
    for i in range(1, N + 1):
        for j in range(0, K + 1):
            p = i % 2  # 获得滚动数组当前索引 k
            if i == 1:
                if j >= Weight[1]:
                    maxValue[p][j] = Value[1]
            else:
                maxValue[p][j] = maxValue[p ^ 1][j]  # 获得滚动数组逻辑上的“上一行”
                if j >= Weight[i]:
                    maxValue[p][j] = max(maxValue[p][j], 
                        maxValue[p^1][j - Weight[i]] + Value[i])
        # print("i = {}, p = {}\n{}".format(i, p, maxValue))
    # Output
    print(np.array(maxValue))
    return maxValue[p][K]

def DP_bag01_dim1(N, Weight, Value, K):
    # Init
    maxValue = np.full(K + 1, 0)
    # Calc
    for i in range(1, N + 1):
        for j in range(K, Weight[i - 1] -1, -1):
            maxValue[j] = max(maxValue[j], 
                maxValue[j - Weight[i - 1]] + Value[i - 1])
        print("i = {}, maxValue = {}".format(i, maxValue))
    # Output
    return maxValue[K]




N, K = 5, 10
Weight = [2, 2, 6, 5, 4]
Values = [6, 3, 5, 4, 6]

N, K = 5, 11
Weight = [1, 2, 5, 6, 7]
Values = [1, 6, 18, 22, 28]

N, K = 4, 10
Weight = [10, 3, 4, 5]
Values = [3, 4, 6, 7]

'''
ans = DP_bag01_dim2(N, Weight, Values, K)
print("Answer: ", ans)

ans = DP_bag01_dim1roll(N, Weight, Values, K)
print("Answer: ", ans)

ans = DP_bag01_dim1(N, Weight, Values, K)
print("Answer: ", ans)
'''



# 2). 完全背包问题


def PkgCompleteBase(N, Weight, Values, K):
    M = np.full([5, K + 1], 0)  # 2^N
    for i in range(1, N + 1):
        for j in range(K + 1):
            tt = j // Weight[i - 1]
            if i == 1:
                if tt >= 1:
                    M[i, j] = tt * Values[i - 1]
            else:
                M[i, j] = M[i - 1, j]
                if tt >= 1:
                    mm = 0  # 对于i个物品，进行j/W[i]次比较得到最大值，而01背包中只需要进行1次比较
                    for k in range(1, tt + 1):
                        ss = M[i - 1, j - k * Weight[i -1]] + k * Values[i -1]
                        if ss > mm:
                            mm = ss
                    M[i, j] = max(M[i, j], mm) 
                    # print("i = {}, tt = {}, mm = {}, ss = {}".format(i, tt, mm, ss))
            # print("i = {}, tt = {}".format(i, tt))
        print("i = {}, M = \n{}".format(i, M[1:]))
    return M[N][K]

def PkgCompleteDim1(N, Weight, Values, K):
    M = np.full(K + 1, 0)
    for i in range(1, N + 1):
        # 正序遍历，01背包是逆序遍历
        for j in range(Weight[i - 1], K + 1):
            M[j] = max(M[j], M[j - Weight[i - 1]] + Values[i - 1])
        print("i = {}, M = \n{}".format(i, M))
    return M[K]


N, K = 4, 10
Weight = [10, 3, 4, 5]
Values = [3, 4, 6, 7]

'''
ans = PkgCompleteBase(N, Weight, Values, K)
print("ans = ", ans)
ans = PkgCompleteDim1(N, Weight, Values, K)
print("ans = ", ans)
'''



# 3). 多重背包问题


def PkgMultipleBase(N, Weight, Values, K, Mounts):
    maxValue = np.full([N + 1, K + 1], 0)
    for i in range(1, N + 1):
        for j in range(K + 1):
            item_num = j // Weight[i -1]
            # if item_num >= 1:
            #     print("i/j={:1d}/{:2d}: num= {}".format(i, j, item_num))
            #   #
            if i == 1:
                if item_num >= 1:
                    maxValue[i, j] = item_num * Values[i -1]
            else:
                item_max = 0
                # 多重背包与完全背包的区别只在内循环这里
                item_lim = min(item_num, Mounts[i -1])
                # if item_lim >= 1:
                #     print("{:9s} num= {}, lim= {}, max= {}".format(
                #         '', item_num, item_lim, item_max))
                #   #
                for k in range(1, item_lim + 1):
                    item_tmp = maxValue[i -1, j - k* Weight[i -1]] + k* Values[i -1]
                    if item_tmp > item_max:
                        item_max = item_tmp
                    # print("{:9s} {:7s} k={}, tmp= {}, max= {}".format(
                    #     '', '', k, item_tmp, item_max))
                maxValue[i, j] = max(maxValue[i - 1, j], item_max)
            #   #
            # if item_num >= 1:
            #     print("i/j={:1d}/{:2d}: {:7s} {:7s} {:7s} maxValue[i]= {}".format(
            #         i, j, '', '', '', maxValue[i]))
        print("i = {}, M = \n{}\n".format(i, maxValue[1:]))
    return maxValue[N, K]

"""
def PkgMultipleDim1(N, Weight, Values, K, Mounts):
    M = np.full(K +1, 0)
    for i in range(1, N + 1):
        item_lim = min(K // Weight[i -1], Mounts[i -1])
        item_cnt = item_lim * Weight[i -1]
        item_itr = 1
        j = 0
        # item_lim = min(K, Mounts[i -1] * Weight[i -1])
        for j in range(Weight[i -1], K + 1):
            item_itr = j // Weight[i -1]
            if item_itr <= item_lim:
                M[j] = max(M[j], M[j - Weight[i -1]] + Values[i -1])
            else:
                M[j] = max(M[j], M[j] + Values[i -1])
        print("i= {:1d}, j= {:2d}, lim= [{:2d}, {:2d}], maxValue= {}".format(i, j, Weight[i-1], item_lim, M))
    return M[K]
"""


N, K = 4, 10
Weight = [10, 3, 4, 5]
Values = [3, 4, 6, 7]
Mounts = [5, 1, 2, 1]

'''
ans = PkgMultipleBase(N, Weight, Values, K, Mounts)
print("ans = ", ans)
ans = PkgMultipleDim1(N, Weight, Values, K, Mounts)
print("ans = ", ans)
'''



#==================================













