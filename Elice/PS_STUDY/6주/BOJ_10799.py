# 10799번 쇠막대기

bar = list(input())
stack = []
result = 0

for i in range(len(bar)):
    if bar[i] == '(':
        stack.append('(')
    else:  # )인 경우
        if bar[i - 1] == '(':
            stack.pop()
            result += len(stack)
        else:
            stack.pop()
            result += 1

print(result)