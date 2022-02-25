# 9012번 괄호

n = int(input())

for _ in range(n):
    stack = []
    input_str = list(input())

    for s in input_str:
        if s == "(":
            stack.append(s)
        else:
            if len(stack) == 0:
                stack.append(-1)
                break
            else:
                stack.pop()
    if len(stack) > 0:
        print("NO")
    else:
        print("YES")