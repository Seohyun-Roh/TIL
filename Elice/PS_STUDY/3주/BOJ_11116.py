# 11116번 교통량

import sys

input = sys.stdin.readline

n = int(input())
for _ in range(n):
    m = int(input())
    result = 0

    left_box = list(map(int, input().split()))
    right_box = list(map(int, input().split()))

    for i in range(m):
        tmp = left_box[i] + 1000
        if tmp in right_box and tmp + 500 in right_box:
            idx_r = right_box.index(tmp)
            if idx_r < right_box.index(tmp + 500):
                result += 1

    print(result)
