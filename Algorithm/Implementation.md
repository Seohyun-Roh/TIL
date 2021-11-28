# 구현(Implementation)

머리속에 있는 알고리즘을 소스 코드로 바꾸는 과정.  
-> 시뮬레이션, 완전 탐색, 구현

시뮬레이션 및 완전 탐색 문제에서는 2차원 공간에서의 방향 벡터가 자주 활용됨.

dx = [0, -1, 0, 1]
dy = [1, 0, -1, 0]

### <문제> 상하좌우 (시뮬레이션 유형)

여행가 A가 n\*n 정사각형 공간 안에 있고, 가장 왼쪽 위가 (1,1), 가장 오른쪽 아래가 (n,n)이다.  
A는 상하좌우로 움직일 수 있고 시작좌표는 항상 (1,1)이다.  
계획서에 L, R, U, D 중 하나의 문자가 적혀있고, A가 최종적으로 도착할 지점의 좌표 (X, Y)를 출력해라.

입력
첫번째 줄: 공간의 크기 n (1 <= n <= 100)
두번째 줄: 여행가가 이동할 계획서 내용 (1 <= 이동횟수 <= 100)

출력
최종적으로 도착할 좌표 (X, Y)를 공백을 기준으로 구분해 출력.

```python
n = int(input())
plans = list(map(str, input().split()))

x, y = 1, 1
dx = [0, 0, -1, 1]
dy = [-1, 1, 0, 0]
moves = ['L', 'R', 'U', 'D']

for plan in plans:
    idx = moves.index(plan)
    tmp_x = x + dx[idx]
    tmp_y = y + dy[idx]
    if (1 <= tmp_x <= n) and (1 <= tmp_y <= n):
        x = tmp_x
        y = tmp_y

print(x, y, end=' ')
```

---

## [ 구현 및 시뮬레이션 문제 추천 ]

### 브론즈/실버 난이도

- [ ] https://www.acmicpc.net/problem/10798
- [ ] https://www.acmicpc.net/problem/2490
- [ ] https://www.acmicpc.net/problem/2884
- [ ] https://www.acmicpc.net/problem/3048
- [ ] https://www.acmicpc.net/problem/2980
- [ ] https://www.acmicpc.net/problem/1063
- [ ] https://www.acmicpc.net/problem/8979
- [ ] https://www.acmicpc.net/problem/2563

### 골드 난이도

- [ ] https://www.acmicpc.net/problem/14500
- [ ] https://www.acmicpc.net/problem/14890
- [ ] https://www.acmicpc.net/problem/17837
- [ ] https://www.acmicpc.net/problem/15683
- [ ] https://www.acmicpc.net/problem/17144
- [ ] https://www.acmicpc.net/problem/15685
- [ ] https://www.acmicpc.net/problem/14499
- [ ] https://www.acmicpc.net/problem/14891
- [ ] https://www.acmicpc.net/problem/15686
- [ ] https://www.acmicpc.net/problem/2563
- [ ] https://www.acmicpc.net/problem/15686
- [ ] https://www.acmicpc.net/problem/14503
- [ ] https://www.acmicpc.net/problem/16235
- [ ] https://www.acmicpc.net/problem/16236

## [ 브루트포스 문제 추천 ]

- [ ] https://www.acmicpc.net/problem/1527
- [ ] https://www.acmicpc.net/problem/1107
- [ ] https://www.acmicpc.net/problem/16943
- [ ] https://www.acmicpc.net/problem/1051
