# coding: utf-8


def subfunc(N, A, i):
    start = A[i]
    gap = A[i+1] - A[i]
    num = 1
    for j in range(i+2, N):
        if A[j] - A[j-1] == gap:
            num += 1
        else:
            break
    return num

def func(N, A):
    num_max = 0
    for i in range(N-1):
        num_iter = subfunc(N, A, i)
        if num_iter > num_max:
            num_max = num_iter
    return num_max


# 1 <= T <= 100, 
# 0 <= Ai <= 10^9
T = int(input())

for x in range(1, T + 1):
    # 2 <= N <= 2000
    # 2 <= N <= 2* 10^5
    N = int(input())
    A = list(map(int, input().strip().split()))
    y = func(N, A)
    print("Case #{}: {}".format(x, y))


"""
Problem
An arithmetic array is an array that contains at least two integers and the differences between consecutive integers are equal. For example, [9, 10], [3, 3, 3], and [9, 7, 5, 3] are arithmetic arrays, while [1, 3, 3, 7], [2, 1, 2], and [1, 2, 4] are not arithmetic arrays.

Sarasvati has an array of N non-negative integers. The i-th integer of the array is Ai. She wants to choose a contiguous arithmetic subarray from her array that has the maximum length. Please help her to determine the length of the longest contiguous arithmetic subarray.

Input
The first line of the input gives the number of test cases, T. T test cases follow. Each test case begins with a line containing the integer N. The second line contains N integers. The i-th integer is Ai.

Output
For each test case, output one line containing Case #x: y, where x is the test case number (starting from 1) and y is the length of the longest contiguous arithmetic subarray.

Limits
Time limit: 20 seconds per test set.
Memory limit: 1GB.
1 ≤ T ≤ 100.
0 ≤ Ai ≤ 109.

Test Set 1
2 ≤ N ≤ 2000.

Test Set 2
2 ≤ N ≤ 2 × 105 for at most 10 test cases.
For the remaining cases, 2 ≤ N ≤ 2000.

Sample

"""

