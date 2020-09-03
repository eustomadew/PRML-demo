# coding: utf8

"""
#coding=utf-8
# 本题为考试单行多行输入输出规范示例，无需提交，不计分。
import sys 
for line in sys.stdin:
    a = line.split()
    print(int(a[0]) + int(a[1]))
"""

"""
#coding=utf-8
# 本题为考试多行输入输出规范示例，无需提交，不计分。
import sys
if __name__ == "__main__":
    # 读取第一行的n
    n = int(sys.stdin.readline().strip())
    ans = 0
    for i in range(n):
        # 读取每一行
        line = sys.stdin.readline().strip()
        # 把每一行的数字分隔后转化成int列表
        values = list(map(int, line.split()))
        for v in values:
            ans += v
    print(ans)
"""



# P3: case accepted 81.82%
#' ''
import sys

K = int(input())  # 1 < K < 1000
N = int(input())  # 1 < N < 1000
W = list(map(int, input().split()))  # 1 < w < 1000
V = list(map(int, input().split()))  # 1 < v < 1000

def bag01(K, Num, Weight, Values):
    matrix = [[0 for j in range(K+1)] for i in range(Num+1)]
    for i in range(1, Num + 1):
        for wc in range(1, K + 1):
            if Weight[i - 1] > wc:
                matrix[i][wc] = matrix[i-1][wc]
            else:
                matrix[i][wc] = max(matrix[i-1][wc],
                    matrix[i-1][wc - Weight[i-1]] + Values[i-1])
    return matrix

matrix = bag01(K, N, W, V)
print(matrix[N][K])
#' ''




# P2: case accepted 70.00%
'''
import sys

def number_of_lakes(lakes, M, N):
    # 'S': water, 'H': solid
    grid = [list(i) for i in lakes]
    #
    def dfs(grid, r, c):
        grid[r][c] = 'H'
        for x, y in [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]:
            if 0 <= x < M and 0 <= y < N and grid[x][y] == 'S':
                dfs(grid, x, y)
    #   #   #   #
    num_lake = 0
    for r in range(M):
        for c in range(N):
            if grid[r][c] == 'S':
                num_lake += 1
                dfs(grid, r, c)
    return num_lake

# 0 < M,N < 1000
M, N = map(int, input().split(','))
lakes = []
for _ in range(M):
    lakes.append(input().strip())

ans = number_of_lakes(lakes, M, N)
print(ans)
'''
'''
import sys

def count_lake(lakes, M, N):
    def dfs(grid, r, c):
        grid[r][c] = 'H'
        for x, y in [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]:
            if 0 <= x < M and 0 <= y < N and grid[x][y] == 'S':
                dfs(grid, x, y)
    #   #   #   #
    grid = [list(i) for i in lakes]
    num_lake = 0
    for r in range(M):
        for c in range(N):
            if grid[r][c] == 'S':
                num_lake += 1
                dfs(grid, r, c)
    return num_lake

while True:
    M, N = sys.stdin.readline().strip().split(',')
    M, N = int(M), int(N)
    lakes = []
    for _ in range(M):
        lakes.append(sys.stdin.readline().strip())
    ans = count_lake(lakes, M, N)
    print(ans)
'''




# P1:
#     case accept 30.00%  # (without else)
#     case accept 80.00%  # (else part)
"""

import sys

def count_candies(N, Sn):
    candy = [None, {}, {}]  # candy1, candy2 = {}, {}
    person = [None, 0, 0]  # person1, person2 = 0, 0
    for loc, (num, clr) in enumerate(Sn):
        if num not in candy[clr]:
            candy[clr][num] = [loc + 1]
        else:
            candy[clr][num].append(loc + 1)
        person[clr] += 1
    return candy, person

def pick_sing_candy(sing_cand, sing_pers):
    locs = sorted(list(sing_cand.keys()))
    locs = locs[-3 :]
    anum = []
    aloc = []
    for k in locs:
        v = sing_cand[k]
        aloc.extend(v)
        anum.extend([k for _ in v])
    return locs, anum, aloc

def output_candy(anum, aloc):
    aloc = ' '.join(map(str, aloc[-3 :]))
    anum = sum(anum[-3 :])
    return anum, aloc

N = int(input())  # N <= 1024
Sn = []
for locat in range(1, N + 1):
    # number, colors = input().strip().split()
    # number, colors = int(number), int(colors)
    Sn.append(list(map(int, input().strip().split())))

candy, person = count_candies(N, Sn)
if person[1] < 3:
    locs, anum, aloc = pick_sing_candy(candy[2], person[2])
    print(' '.join(map(str, aloc[-3 :])))
    print(2)
    print(sum(anum[-3 :]))
elif person[2] < 3:
    locs, anum, aloc = pick_sing_candy(candy[1], person[1])
    print(' '.join(map(str, aloc[-3 :])))
    print(1)
    print(sum(anum[-3 :]))
else:
    _, num1, loc1 = pick_sing_candy(candy[1], person[1])
    _, num2, loc2 = pick_sing_candy(candy[2], person[2])
    # loc1 = ' '.join(map(str, loc1[-3 :]))
    # loc2 = ' '.join(map(str, loc2[-3 :]))
    num1, loc1 = output_candy(num1, loc1)
    num2, loc2 = output_candy(num2, loc2)
    if num1 > num2:
        print('{}\n{}\n{}'.format(loc1, 1, num1))
    elif num2 > num1:
        print('{}\n{}\n{}'.format(loc2, 2, num2))
    else:
        if loc1[0] <= loc2[0]:
            print('{}\n{}\n{}'.format(loc1, 1, num1))
        else:
            print('{}\n{}\n{}'.format(loc2, 2, num2))
"""
