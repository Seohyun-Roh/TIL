# 4358번 생태학

import sys
input = sys.stdin.readline

total = 0
trees = {}

while True:
    t = input().rstrip()
    if not t:  # 빈 입력이 들어 오면 입력 멈춤
        break

    # 입력값이 있으면 가져온 후 +1한 값 저장. 없으면 기본값 0에 +1해서 저장.
    trees[t] = trees.get(t, 0) + 1
    total += 1

trees_keys = sorted(trees.keys())

for key in trees_keys:
    print('%s %.4f' % (key, trees[key] / total * 100))