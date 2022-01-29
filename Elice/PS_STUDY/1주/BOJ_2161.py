# 2161번 카드1
import sys

input = sys.stdin.readline
n = int(input())
cards = []
for i in range(1, n + 1):
    cards.append(i)

trash = []

while len(cards) != 1:
    trash.append(cards.pop(0))
    cards.append(cards.pop(0))
for i in range(n - 1):
    print(trash[i], end=' ')
print(cards[0])