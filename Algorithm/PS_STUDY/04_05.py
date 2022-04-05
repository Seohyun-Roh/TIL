'''
# 1541번 잃어버린 괄호

val = input().split('-')
res = 0

for i in range(len(val)):
    tmp = sum(list(map(int, val[i].split('+'))))
    if i == 0:
        res = tmp
    else:
        res -= tmp

print(res)

# 2217번 로프

import sys

input = sys.stdin.readline

n = int(input())
rope = [int(input()) for _ in range(n)]
max_w = 0  # 최대 중량

rope.sort()
for i in range(n):
    tmp = rope[i] * (n - i)
    if tmp > max_w:
        max_w = tmp

print(max_w)
'''

# 13458번 시험 감독
import math

n = int(input())
a = list(map(int, input().split()))
b, c = map(int, input().split())

supervisior = 0

for people in a:
    supervisior += 1
    tmp = people - b
    if tmp > 0:
        supervisior += math.ceil(tmp / c)

print(supervisior)
