# 22.02.02
# 이코테 알고리즘 유형별 기출 - 그리디, 구현문제

# p.315 볼링공 고르기

n, m = map(int, input().split())
balls = list(map(int, input().split()))

cnt = [0] * m
res = 0

for i in range(m):
    cnt[i] = balls.count(i + 1)
print(cnt)

for i in range(m):
    n -= cnt[i]
    res += cnt[i] * n
    print(res, cnt[i], n)

print(res)

# p. 321 럭키 스트레이트

score = input()
mid = len(score) // 2
left = int(score[:mid])
right = int(score[mid:])
sum1 = sum2 = 0

for i in range(mid):
    sum1 += left % 10
    left //= 10
    sum2 += right % 10
    right //= 10

if sum1 == sum2:
    print("LUCKY")
else:
    print("READY")

# p. 322 문자열 재정렬

input_str = input()
alpha = []
sum = 0

for x in input_str:
    if 65 <= ord(x) <= 90:  # 대문자
        alpha.append(x)
    else:   # 숫자
        sum += int(x)

alpha.sort()

for a in alpha:
    print(a, end='')
print(sum)