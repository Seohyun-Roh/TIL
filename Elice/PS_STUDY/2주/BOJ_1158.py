# 1158번 요세푸스 문제

n, k = map(int, input().split())

people = list(range(1, n + 1))
res = []
rmv = k - 1

while len(people) > 1:
    if rmv < len(people):
        res.append(people.pop(rmv))
        rmv += k - 1
    else:
        rmv = rmv % len(people)
        res.append(people.pop(rmv))
        rmv += k - 1
        
print("<", end='')
for x in res:
    print(x, ", ", end='', sep='')
print(people[0], ">", sep='')