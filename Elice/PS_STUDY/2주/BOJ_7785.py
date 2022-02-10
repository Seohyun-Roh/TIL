# 7785번 회사에 있는 사람

import sys
input = sys.stdin.readline

n = int(input())
people = dict()

for _ in range(n):
    name, log = input().split()

    if log == "enter":
        people[name] = 1
    elif log == "leave":
        del people[name]

people = sorted(people.keys(), reverse=True)
for p in people:
    print(p)