#coding=utf-8
# Huawei 2020,09.02  Software Test

'''
# 本题为考试单行多行输入输出规范示例，无需提交，不计分。
import sys 
for line in sys.stdin:
    a = line.split()
    print(int(a[0]) + int(a[1]))
'''

'''
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
'''




# Problem 1
"""
import sys
lines = sys.stdin.readlines()
for read in lines:
    a, b = map(int, read.split())
    print(a + b)
"""

# Problem 2

import sys
lines = sys.stdin.readlines()

def dereplicate(a):
    s = list(a)
    d = {}
    for i, k in enumerate(a):
        if k not in d:
            d[k] = 1
        else:
            s[i] = ''
    return ''.join(s)

for read in lines:
    output = dereplicate(read)
    print(output)

