# 11279번 최대 힙

import sys
import heapq  # heapq: 최소 힙으로 구현되어 있음.

input = sys.stdin.readline

n = int(input())
max_heap = []

for _ in range(n):
    x = int(input())
    if x == 0:
        if len(max_heap) != 0:
            print(heapq.heappop(max_heap)[1])
        else:
            print(0)
    heapq.heappush(max_heap, (-x, x))