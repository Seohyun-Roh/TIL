# 4158번 CD

import sys
input = sys.stdin.readline

while True:
    n, m = map(int, input().split())
    
    if n == 0 and m == 0:
        break

    cd_1 = set([])
    cd_2 = set([])

    for _ in range(n):
        cd_1.add(int(input()))
    for _ in range(m):
        cd_2.add(int(input()))

    print(len(cd_1 & cd_2))