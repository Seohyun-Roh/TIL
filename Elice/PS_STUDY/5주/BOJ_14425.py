# 14425번 문자열 집합

n, m = map(int, input().split())
numbers = []
count = 0

for _ in range(n):
    numbers.append(input())

for _ in range(m):
    s = input()
    if s in numbers:
        count += 1
        
print(count)