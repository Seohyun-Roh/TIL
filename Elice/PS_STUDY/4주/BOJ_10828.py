# 10828번 스택

import sys

input = sys.stdin.readline

n = int(input())
stack = []

for _ in range(n):
    tmp = input().split()

    if tmp[0] == "push":
        stack.append(tmp[1])
    elif tmp[0] == "pop":
        if len(stack) == 0:
            print(-1)
        else:
            print(stack.pop())
    elif tmp[0] == "size":
        print(len(stack))
    elif tmp[0] == "empty":
        if len(stack) == 0:
            print(1)
        else:
            print(0)
    else:
        if len(stack) == 0:
            print(-1)
        else:
            print(stack[len(stack) - 1])