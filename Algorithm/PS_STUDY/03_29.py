'''
# 11047번 동전 0
n, k = map(int, input().split())
coins = []
res = 0

for i in range(n):
    coins.append(int(input()))

coins.sort(reverse=True)

for coin in coins:
    if coin > k:
        continue
    res += k // coin
    k %= coin

print(res)


# 11000번 강의실 배정
import heapq
import sys

input = sys.stdin.readline

n = int(input())
lectures = []

for _ in range(n):
    lectures.append(list(map(int, input().split())))

lectures.sort(key=lambda x: x[0])

tmp = []
heapq.heappush(tmp, lectures[0][1])

for i in range(1, n):
    if tmp[0] > lectures[i][0]:
        heapq.heappush(tmp, lectures[i][1])
    else:
        heapq.heappop(tmp)
        heapq.heappush(tmp, lectures[i][1])

print(len(tmp))


# 11399번 ATM

n = int(input())
p = list(map(int, input().split()))

p.sort()
res = 0

for i in range(n):
    res += sum(p[:i+1])

print(res)
'''

# 1931번 회의실 배정
n = int(input())
meetings = [list(map(int, input().split())) for _ in range(n)]

meetings.sort(key=lambda x: (x[1], x[0]))
tmp = 0
cnt = 0

for start, end in meetings:
    if tmp <= start:
        cnt += 1
        tmp = end

print(cnt)
