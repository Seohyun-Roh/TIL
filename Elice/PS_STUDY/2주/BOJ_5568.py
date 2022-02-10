# 5568번 카드 놓기

import itertools

n = int(input())
k = int(input())
cards = [input() for _ in range(n)]
numbers = set()

for x in itertools.permutations(cards, k):
    numbers.add(x)

print(numbers)