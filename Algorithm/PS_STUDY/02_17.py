# 22.02.17
# 이코테 정렬 예제

# p.178 위에서 아래로

# n = int(input())
# numbers = []
#
# for _ in range(n):
#     numbers.append(int(input()))
#
# numbers.sort(reverse=True)
#
# for i in range(n):
#     print(numbers[i], end=' ')

# p. 180 성적이 낮은 순서로 학생 출력하기

# n = int(input())
# students = {}
#
# for _ in range(n):
#     name, grade = input().split()
#     grade = int(grade)
#     students[name] = grade
#
# students = sorted(students.items(), key=lambda x: x[1])
#
# for i in range(n):
#     print(students[i][0], end=' ')

# p. 182 두 배열의 원소 교체

n, k = map(int, input().split())

list_a = list(map(int, input().split()))
list_b = list(map(int, input().split()))

list_a.sort()
list_b.sort(reverse=True)

for i in range(k):
    if list_a[i] < list_b[i]:
        list_a[i], list_b[i] = list_b[i], list_a[i]
    else:
        break

print(sum(list_a))