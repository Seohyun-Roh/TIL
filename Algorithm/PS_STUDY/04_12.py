'''
# 4796번 캠핑

case = 1

while True:
    L, P, V = map(int, input().split())
    if L == 0 and P == 0 and V == 0:
        break
    result = V // P * L
    result += min(V % P, L)
    print("Case ", case, ": ", result, sep="")
    case += 1


# 12845번 모두의 마블

n = int(input())
cards = list(map(int, input().split()))

cards.sort()
gold = 0

while len(cards) > 1:
    gold += cards[-2] + cards[-1]
    cards[-2] = cards[-1]
    cards.pop()

print(gold)
'''

# 1700번 멀티탭 스케줄링
# 다시 풀어보기

# n, k = map(int, input().split())
# appliance = list(map(int, input().split()))
# plug = []
# result = 0
#
# for i in range(len(appliance)):
#     if appliance[i] in plug: # 전자제품이 이미 꽂혀있으면 continue
#         continue
#
#     if len(plug) < n:  # 멀티탭이 비어있으면 꽂음
#         plug.append(appliance[i])
#         continue
#
#     target = -1
#     idx = -1
#     for i in range(n):
#         if plug[i] not in appliance:  # 다음에 사용되지 않는 전자제품이면 뽑음.
#             plug.pop(i)
#             plug.append(appliance[i])
#             result += 1
#             break
#         else:  # 다음에 사용되는 전자제품일 경우
#             # 나중에 있는 것을 뽑아야 함.
#             if idx < appliance[i:].index(plug[i]):
#                 target = plug[i]
#                 idx = appliance[i:].index(plug[i])
#
#     if len(plug) == n and idx != -1:
#         plug.remove(appliance[idx])
#         plug.append(appliance[i])
#         result += 1
# print(result)

# 1969번 DNA
# 다시 풀어보기
# n, m = map(int, input().split())
# dna = [{} * m]
#
# for _ in range(n):
#     count_dna = {}
#     tmp = input().split()
#     for i in range(len(tmp)):
#         dna[i][tmp[i]] = 'a'
# print(dna)

# 13305번 주유소

n = int(input())
roads = list(map(int, input().split()))
costs = list(map(int, input().split()))

minVal = costs[0]
sum = 0

for i in range(n - 1):
    if minVal > costs[i]:
        minVal = costs[i]
    sum += (minVal * roads[i])

print(sum)