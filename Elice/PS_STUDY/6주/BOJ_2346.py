# 2346번 풍선 터뜨리기

from collections import deque

n = int(input())
papers = list(map(int, input().split()))
deq = deque(list(range(1, n + 1)))

while deq:
    idx = deq.popleft()
    print(idx, end=' ')

    tmp = papers[idx - 1]

    if tmp > 0:
        deq.rotate((tmp - 1) * (-1))
    else:
        deq.rotate(tmp * (-1))

###

# deq = deque([1, 2, 3, 4, 5])

# # rotate 양수-> 오른쪽으로 회전.
# deq.rotate(2)
# print(deq)
# # deque([4, 5, 1, 2, 3])

# deq2 = deque([1, 2, 3, 4, 5])
# 
# # rotate 음수-> 왼쪽으로 회전.
# deq2.rotate(-2)
# print(deq2)
# # deque([3, 4, 5, 1, 2])