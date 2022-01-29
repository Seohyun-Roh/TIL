# 12605번 단어순서 뒤집기

n = int(input())

for i in range(n):
    word = list(input().split(' '))
    print("Case #", i + 1, ": ", sep='', end='')
    for j in range(len(word) - 1, -1, -1):
        print(word[j], end=' ')