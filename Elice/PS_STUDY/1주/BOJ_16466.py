# 16466번 콘서트

import sys

input = sys.stdin.readline
n = int(input())
tickets = sorted(list(map(int, input().split())))

for i in range(1, n + 1):
    if tickets[i - 1] != i:
        print(i)
        sys.exit()

print(n + 1)