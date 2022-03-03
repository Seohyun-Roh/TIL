# 22.03.02 이코테 이진 탐색 예제
# p. 197 부품 찾기

# n개 부품. m개 종류 부품을 대량 구매. 부품이 모두 있는지 확인
# 부품 번호. 있으면 yes

# def binary_search(array, target):
#     start = 0
#     end = len(array)-1
#     while start <= end:
#         mid = (start + end) // 2
#         if array[mid] == target:
#             return mid
#         elif array[mid] > target:
#             end = mid - 1
#         else:
#             start = mid + 1
#     return None
#
#
# n = int(input())
# array = list(map(int, input().split()))
# m = int(input())
# req = list(map(int, input().split()))
#
# array.sort()
#
# for i in req:
#     if binary_search(array, i):
#         print("yes", end=' ')
#     else:
#         print("no", end=' ')

# p. 201 떡볶이 떡 만들기

n, m = map(int, input().split())
array = list(map(int, input().split()))

start = 0
end = max(array)

res = 0
while start <= end:
    tmp = 0
    mid = (start + end) // 2
    for x in array:
        if x > mid:
            tmp += x - mid
    if tmp < m:  # 더 자른 경우-> 높이 낮춰야 함.
        end = mid - 1
    else:  # 덜 자른 경우-> 높이 높여야 함
        start = mid + 1
        res = mid
print(res)
