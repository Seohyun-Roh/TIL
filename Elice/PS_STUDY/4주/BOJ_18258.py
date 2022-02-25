# 18258번 큐 2

import sys

input = sys.stdin.readline

n = int(input())
queue = []
pt = 0

for _ in range(n):
    tmp = input().split()
    if tmp[0] == "push":
        queue.append(tmp[1])
    elif tmp[0] == "pop":
        if len(queue) - pt == 0:
            print(-1)
        else:
            print(queue[pt])
            pt += 1
    elif tmp[0] == "size":
        print(len(queue) - pt)
    elif tmp[0] == "empty":
        if len(queue) - pt == 0:
            print(1)
        else:
            print(0)
    elif tmp[0] == "front":
        if len(queue) - pt == 0:
            print(-1)
        else:
            print(queue[pt])
    else:
        if len(queue) - pt == 0:
            print(-1)
        else:
            print(queue[len(queue) - 1])