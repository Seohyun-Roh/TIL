# 23253번 자료구조는 정말 최고야

import sys

input = sys.stdin.readline

n, m = map(int, input().split())

result = "Yes"

for _ in range(m):
    k = int(input())
    books = list(map(int, input().split()))
    sorted_books = sorted(books, reverse=True)
    if books != sorted_books:
        result = "No"

print(result)