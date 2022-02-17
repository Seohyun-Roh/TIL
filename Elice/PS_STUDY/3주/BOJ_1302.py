# 1302번 베스트셀러

n = int(input())
books = {}

for _ in range(n):
    book = input()
    books[book] = books.get(book, 0) + 1

max_count = max(books.values())
best_seller = []

for book, count in books.items():
    if count == max_count:
        best_seller.append(book)

best_seller.sort()

print(best_seller[0])