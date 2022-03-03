# 1620번 나는야 포켓몬 마스터 이다솜

import sys

input = sys.stdin.readline

n, m = map(int, input().rstrip().split())
pokemon = {}

for i in range(1, n + 1):
    tmp = input().rstrip()
    pokemon[i] = tmp
    pokemon[tmp] = i

for i in range(m):
    test = input().rstrip()
    if test.isdigit():  # isdigit() -> 문자열이 숫자 형태면 True 반환
        print(pokemon[int(test)])
    else:
        # 값을 이용해서 key값 찾는 방법 -> 시간 초과
        # tmp = [key for key, val in pokemon.items() if val == test]
        # print(tmp)
        print(pokemon[test])