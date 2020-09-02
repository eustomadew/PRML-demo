# coding: utf-8
# ref:
# https://www.nowcoder.com/test/question/3897c2bcc87943ed98d8e0b9e18c4666?pid=260145&tid=36842386



# Problem 1

'''
def maximum_score(A, low=0, high=1):
    pass

N, M = map(int, input().split())
scores = list(map(int, input().split()))
for _ in range(M):
    C, A, B = input().split()
    A, B = int(A), int(B)
    if C == 'U':
        scores[A - 1] = B
    elif C == 'Q':
        if A > B:
            B, A = A, B
        answer = max(scores[A - 1: B])
        print(answer)
'''

"""
9 10
28 49 11 35 40 17 57 4 6
Q 9 9
U 9 79
Q 9 5
Q 4 8
U 2 27
U 8 40
U 4 77
U 7 71
U 4 44
U 8 51

ANSWER
6
79
57
"""



# Problem 2
# python Huawei2016.py < Huawei2016.txt
# E:\V1R2\product\fpgadrive.c 1325

"""
import sys
records = sys.stdin.readlines()
# print(records)

records = [i.strip() for i in records]  # [:8]]
stats = {}
for temp in records:
    file, line = temp.split()
    line = int(line)
    fold = file.split('\\')
    path = '\\'.join(fold[: -1])
    name = fold[-1]
    #
    if file not in stats:
        stats[file] = {line: 1}
    elif line not in stats[file]:
        stats[file][line] = 1
    else:
        stats[file][line] += 1
print("{}\n".format(stats))

'''
for k1, v1 in stats.items():
    x = k1.split('\\')[-1]
    for k2, v2 in v1.items():
        print(x, k2, v2)
'''

maxi = -1
for k, v in stats.items():
    maxi = max(maxi, max(v.values()))
# output = [{} for _ in range(maxi + 1)]
output = {i: [] for i in range(maxi + 1)}
for k1, v1 in stats.items():
    x = k1.split('\\')[-1]
    for k2, v2 in v1.items():
        st = '{} {} {}'.format(x, k2, v2)
        output[v2].append(st)

numi = 0
for k in range(maxi, 0, -1):
    for v in output[k]:
        print(v)
        numi += 1
        if numi >= 8:
            break
    if numi >= 8:
        break
"""



# Problem 3

import sys
cards = sys.stdin.readlines()
cards = [i.strip() for i in cards]
# print(cards)

def is_duiwang(sing):
    if 'joker' in sing and 'JOKER' in sing:
        return True
    return False

def is_same(sing):
    sing = sing.split()
    if len(set(sing)) == 1:
        if len(sing) == 4:
            return 'zhadan'
        elif len(sing) == 3:
            return 'sange'
        elif len(sing) == 2:
            return 'duizi'
        elif len(sing) == 1:
            return 'gezi'
    return False

def is_shunzi(sing):
    sing = sing.split()
    if len(set(sing)) == 5 and len(sing) == 5:
        sing = ' '.join(sing)
        if sing == '3 4 5 6 7':
            return True
        elif sing == '4 5 6 7 8':
            return True
        elif sing == '5 6 7 8 9':
            return True
        elif sing == '6 7 8 9 10':
            return True
        elif sing == '7 8 9 10 J':
            return True
        elif sing == '8 9 10 J Q':
            return True
        elif sing == '9 10 J Q K':
            return True
        elif sing == '10 J Q K A':
            return True
        elif sing == 'J Q K A 2':
            return True
    return False



def compare(card):
    A, B = card.split('-')
    if is_duiwang(A) or is_duiwang(B):
        return A if is_duiwang(A) else B
    sa, sb = is_same(A), is_same(B)
    if sa == 'zhadan' or sb == 'zhadan':
        if sa == 'zhadan' and 'sb' == 'zhadan':
            return A if A[0] > B[0] else B
        else:
            return A if sa=='zhadan' else B
    return card

for x in cards:
    y = compare(x)
    print(y)

