# 17608번 막대기

import sys

input = sys.stdin.readline
n = int(input())

sticks = []

for i in range(n):
    sticks.append(int(input()))

cnt = 1
max_num = sticks[-1]

for i in range(n - 1, -1, -1):
    if max_num < sticks[i]:
        cnt += 1
        max_num = sticks[i]

print(cnt)