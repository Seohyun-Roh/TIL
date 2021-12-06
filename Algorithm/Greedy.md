# Greedy Algorithm

그리디 알고리즘(탐욕법)은 현재 상황에서 지금 당장 좋은 것만 고르는 방법을 의미함.

일반적인 상황에서 그리디 알고리즘은 최적의 해를 보장할 수 없을 때가 많음.  
하지만 코딩 테스트의 대부분의 그리디 문제는 탐욕법으로 얻은 해가 최적의 해가 되는 상황에서 이를 추론할 수 있어야 풀리도록 출제됨.

그리디 알고리즘 문제에서는 문제 풀이를 위한 최소한의 아이디어를 떠올리고 이가 정당한지 검토할 수 있어야 함.

### <문제> 1이 될 때까지

어떤 수 n이 1이 될 때까지 아래의 과정 중 하나를 반복적으로 수행하려 한다.

- 1. n에서 1을 뺀다.
- 2. n을 k로 나눈다.

n, k가 주어질 때 n이 1이 될 때까지 1번 혹은 2번의 과정을 수행해야 하는 최소 횟수를 구하는 프로그램을 작성하시오.

```python
n, k = map(int, input().split())

result = 0

while True:
    tmp = (n // k) * k
    result += (n - tmp)
    n = tmp
    if n < k:  # 더 나눌 수 없을 때 반복문 탈출
        break
    result += 1  # k로 나누는 연산 1번
    n //= k  # k로 나누기

result += (n - 1)
print(result)
```

## [문제 추천]

- [x] https://www.acmicpc.net/problem/1700
- [x] https://www.acmicpc.net/problem/2875
- [x] https://www.acmicpc.net/problem/1783
- [ ] https://www.acmicpc.net/problem/11000
- [ ] https://www.acmicpc.net/problem/2217
- [ ] https://www.acmicpc.net/problem/13458
- [x] https://www.acmicpc.net/problem/1946
- [ ] https://www.acmicpc.net/problem/12845
- [ ] https://www.acmicpc.net/problem/2873
- [ ] https://www.acmicpc.net/problem/1744
- [ ] https://www.acmicpc.net/problem/1969
