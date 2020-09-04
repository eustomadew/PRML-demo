=============================
算法 (Algorithms)
=============================


-----------------------------
动态规划
-----------------------------



背包问题
=============================

**refs**:

- 0-1背包问题的动态规划算法: https://zhuanlan.zhihu.com/p/30959069
- 动态规划之01背包问题: https://blog.csdn.net/mu399/article/details/7722810
- 三种背包问题: 01背包、完全背包、多重背包 https://blog.csdn.net/sinat_30973431/article/details/85119871
- 完全背包详解及实现 (包含背包具体物品的求解) https://blog.csdn.net/wumuzi520/article/details/7014830
- 动态规划 详解完全背包与多重背包 https://juejin.im/post/6844904103122845704


给定一组多个 (:math:`N`) 物品，每种物品都有自己的重量 (:math:`w_i`) 和价值 (:math:`v_i`)，在限定的总重量/总容量 (:math:`K`) 内，选择其中若干个 (亦即每种物品可以选 0 个或 1 个)，设计选择方案使得物品的总价值最高。

**形式化为** 
给定正整数 :math:`\{(w_i,v_i)\}_{1\le i\le N}` 和正整数 :math:`K`，求解 0-1 规划问题： 
:math:`\max \sum_{i=1}^N x_i v_i 
\,,\; \text{ s.t. }
\sum_{i=1}^N x_i w_i \le K
\,,\; x_i\in \{0,1\}`

**示例应用:** 
处理器能力有限时间受限，任务很多，如何选择使得总效用最大？


***递推关系***

定义子问题 :math:`P(i,C)` 为: 
在前 :math:`i` 个物品中挑选总重量不超过 :math:`C` 的物品, 每种物品至多只能挑选 1 个, 使得总价值最大; 
这时的最优值记作 :math:`f(i,C)`, 其中 :math:`1\le i\le N`, :math:`1\le C\le K`.

考虑第 :math:`i` 个物品, 无外乎两种可能: 选, 或者不选. 

- 不选的话, 背包的容量不变, 改变为问题 :math:`P(i-1,C)`; 
- 选的话，背包的容量变小，改变为问题 :math:`P(i-1, C-w_i)`. 

最优方案就是比较这两种方案，哪个会更好些: 
:math:`f(i,C) = \max\{ f(i-1,C) ,\, f(i-1, C-w_i) + v_i \}`, 

得到 
:math:`f(i,C) = \begin{cases}
0 \,,&\text{ if } i=0 \,;\\
0 \,,&\text{ if } C=0 \,;\\
f(i-1, C) \,,&\text{ if } w_i > C \,;\\
\max\{ f(i-1,C) ,\, f(i-1, C-w_i) + v_i \} \,,&\text{ otherwise }
\end{cases}`


**动态规划方法**
-----------------------

- **填二维表** 的动态规划方法

    ::

        **Inputs**: N, Weight[1:n], Value[1:n], K

        M = np.full([N+1, K+1], -1)
        for w in range(0, K + 1):
            M[0, w] = 0
        for i in range(1, N + 1):
            M[i, 0] = 0

        for i in range(1, N + 1):
            for C in range(1, K + 1):
                if Weight[i] > C:
                    M[i, C] = M[i - 1, C]
                else:
                    M[i, C] = max(M[i-1, C], M[i-1, C-Weight[i]] + Value[i])

        return M[N, C]

    时间复杂度和空间复杂度都是 :math:`\Omega(NK)`. 
    当 :math:`K > 2^N` 时，复杂度是 :math:`\Omega(N2^N)`. 

- **填一维表** 的动态规划方法

    - 三个问题：

        - 显然, :math:`f(i,C) \ge f(i-1,C)`, :math:`f(i,C) \ge f(i,C-1)`
        - 何时发生 :math:`f(i,C) > f(i, C-1)`?
        - 何时发生 :math:`f(i,C) > f(i-1, C)`?

    - 问题 2: When :math:`f(i,C) > f(i, C-1)`

        一定是发生了"容量扩大后有个新的东西可以放下了"！ 
        所以固定 :math:`i`, 让:math:`C` 变化, 则 :math:`f(i,C)` 一定是"阶梯状"的:

        - 有的 :math:`w` 使得 :math:`f(i,w) > f(i,w-1)`; 
        - 有的 :math:`w` 使得 :math:`f(i,w) = f(i,w-1)`. 

        于是:

        - 对于每一个 :math:`i`, :math:`f(i,C)` 最多只有 :math:`2^i` 个"转折点" —— 因为 :math:`i` 个物品 最多只有 :math:`2^i` 个"选"、"不选"的组合 
        - :math:`f(2,C)` 中 :math:`f(1,C-w_2)+v_2` 那部分的所有 **可能的** "转折点"是由 :math:`f(1,C)` 的每个转折点 :math:`(1,w,v)` 变为 :math:`(2, w+w_2, v+v_2)` ("可能"这个词后面解释); 
        - 推而广之, :math:`f(i+1,C)` 中 :math:`f(i, C-w_{i+1}) +v_{i+1}` 那部分的所有 **可能的** "转折点"就是由 :math:`(i,w,v)` 的每个转折点  变为 :math:`(i+1, w+w_{i+1}, v+v_{i+1})` 的.

        设置 :math:`S^0 = \{(0,0,0)\}`, 则由 :math:`S^i` 得到 :math:`S^{i+1}` 的所有可能的"转折点"为 :math:`\{(i+1, w+w_{i+1}, v+v_{i+1}) |\,  (i,w,v)\in S^i \}`. 

        这时有些问题: 

        1. 超过 :math:`C=11` 的部分可以不用考虑; 
        2. 绿色的圆形里有些"转折点"被湮没了 —— 这就是之前说的"可能"的意思. 

    - 问题 3: When :math:`f(i,C) > f(i-1, C)`



`0 - 1` 背包
-----------------------

**基本实现**

时间复杂度 :math:`O(NK)`, 空间复杂度 :math:`O(NK)` (可优化)

::

    def DP_bag01_dim2(N, Weight, Value, K):
        M = np.full([N +1, K +1], 0)
        for i in range(1, N + 1):
            for C in range(1, K + 1):
                if Weight[i - 1] > C:
                    M[i, C] = M[i - 1][C]
                else:
                    M[i, C] = max(M[i - 1, C],
                                M[i - 1, C - Weight[i-1]] + Value[i-1])
        return M[N, C]

**滚动数组实现**

::

    def DP_bag01_roll(N, Weight, Value, K):
        p = 0
        M = np.full([2, K +1], 0)
        for i in range(1, N + 1):
            for j in range(K + 1):
                p = i % 2
                if i == 1:
                    if j >= Weight[i - 1]:
                        M[p, j] = Value[i]
                else:
                    M[p, j] = M[p ^ 1, j]
                    if j >= Weight[i - 1]:
                        M[p, j] = max(M[p, j],
                                    M[p^1, j - Weight[i-1]] + Value[i-1])
        return M[p][K]

**一维数组实现**

::

    def DP_bag01_dim1(N, Weight, Value, K):
        M = np.full(K +1, 0)
        for i in range(1, N + 1):
            for j in range(K, Weight[i - 1] - 1, -1):
                M[j] = max(M[j], M[j - Weight[i - 1]] + Value[i - 1])
        return M[K]


完全背包
-----------------------

完全背包是在 01 背包的基础上加了个条件 —— 
这 n 种物品都有无限数量可取，问怎样拿方可实现价值最大化。

隐藏条件： 
背包承重量的固定性导致每种最多只能取某个值，即 :math:`K/w_i`，再多就放不下了。
也就是说，对于第 i 种物品，可取 :math:`0,1,2,..., \lfloor \frac{K}{w_i} \rfloor` (向下取整) 件。
而在 01 背包中，对于第 i 种物品，只能取 0,1 件。

**基本实现**

::

    def Pkg_complete_dim2(N, Weight, Values, K):
        M = np.full([N +1, K +1], 0)  # 2^N
        for i in range(1, N + 1):
            for j in range(K + 1):
                item_num = j // Weight[i -1]
                if i == 1:
                    if item_num >= 1:
                        M[i, j] = item_num * Values[i -1]
                else:
                    M[i, j] = M[i - 1, j]
                    if item_num >= 1:
                        item_max = 0
                        # 对于第 i 个物品，进行 j/W[i] 次比较得到最大值，
                        # 而 01 背包中只需要进行 1 次比较
                        for k in range(1, item_num + 1):
                            item_tmp = M[i - 1, j - k * Weight[i-1]] + k * Values[i-1]
                            if item_tmp > item_max:
                                item_max = item_tmp
                        M[i, j] = max(M[i, j], item_max)
        return M[N, K]

**时间优化**

基本实现中的时间复杂度为 :math:`O(NK \sum \frac{K}{w_i}) = O(NK \cdot\max(\frac{K}{w_i}))`

**空间优化**

::

    def Pkg_complete_dim1(N, Weight, Values, K):
        M = np.full(K +1, 0)
        for i in range(1, N + 1):
            # 正序遍历，01 背包中是逆序遍历
            for j in range(Weight[i -1], K + 1):
                M[j] = max(M[j], M[j - Weight[i -1]] + Values[i -1])
        return M[K]


多重背包
-----------------------

多重背包是在 `01` 背包的基础上加了个条件： 
第 :math:`i` 件物品有 :math:`n_i` 件

**基本实现**

如果所有 :math:`n_i` 都满足 :math:`n_i \ge K/w_i`, 即为完全背包问题。
完全背包与多重背包的区别在于物品的个数上界不再是 :math:`K/w_i`, 而是 :math:`n_i` 与 :math:`K/w_i` 中较小的一个。
所以要在完全背包的基本实现之上，再考虑这个上界问题。

::

    def Pkg_multiple_dim2(N, Weight, Values, K, Mounts):
        M = np.full([N +1, K +1], 0)
        for i in range(1, N + 1):
            for j in range(K + 1):
                item_num = j // Weight[i -1]
                if i == 1:
                    if item_num >= 1:
                        M[i, j] = item_num * Values[i -1]
                else:
                    item_max = 0
                    item_lim = min(item_max, Mounts[i -1])
                    # 多重背包与完全背包的区别只在内层循环这里
                    for k in range(1, item_lim + 1):
                        item_tmp = M[i -1, j - k* Weight[i -1]] + k* Values[i -1]
                        if item_tmp > item_max:
                            item_max = item_tmp
                    M[i, j] = max(M[i - 1, j], item_max)
        return M[N, K]

**时间优化** (通过二进制拆分转化为 01 背包问题)

**优先队列实现**


