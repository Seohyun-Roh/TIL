# 2358번 평행선

n = int(input())
point_x = {}
point_y = {}
result = 0

for i in range(n):
    x, y = map(int, input().split())
    if x in point_x:
        point_x[x] += 1
    else:
        point_x[x] = 1
    if y in point_y:
        point_y[y] += 1
    else:
        point_y[y] = 1

for x in point_x.values():
    if x >= 2:
        result += 1
for y in point_y.values():
    if y >= 2:
        result += 1

print(result) 