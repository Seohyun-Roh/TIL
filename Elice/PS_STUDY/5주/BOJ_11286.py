# 11286번 절댓값 힙

import heapq
import sys
input = sys.stdin.readline

n = int(input())
heap = []

for _ in range(n):
    x = int(input())

    if x == 0:
        if len(heap) == 0:
            print(0)
            continue
        print(heapq.heappop(heap)[1])
    else:
        # 절댓값이 작은 값을 기준으로 최소힙 구성
        heapq.heappush(heap, (abs(x), x))