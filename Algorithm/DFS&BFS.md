# DFS(Depth-First Search: 깊이 우선 탐색)

깊은 부분을 우선적으로 탐색하는 알고리즘. 스택이나 재귀함수를 이용해 구현.

1. 탐색 시작 노드를 스택에 삽입한 후 방문 처리를 한다.

2. 스택의 최상단 노드와 인접한 노드 중에 방문하지 않은 노드가 있으면 해당 노드를 스택에 넣고 방문 처리를 한다. 방문하지 않은 노드가 없다면(인접 노드를 모두 방문 했다면) 스택의 최상단 노드에서 꺼낸다.

3. 2의 과정을 수행할 수 없을 때까지 반복한다.

```python
def dfs(graph, v, visited):
    visited[v] = True
    print(v, end=' ')
    for i in graph[v]:
        if not visited[i]:
            dfs(graph, i, visited)

graph = [
    [],
    [2, 3, 8],
    [1, 7],
    [1, 4, 5],
    [3, 5],
    [3, 4],
    [7],
    [2, 6, 8],
    [1, 7]
]

visited = [False] * 9

dfs(graph, 1, visited)
# 1 2 7 6 8 3 4 5
```

# BFS(Breadth-First Search: 너비 우선 탐색)

가까운 노드부터 우선 탐색하는 알고리즘. 큐 자료구조를 이용.  
각 간선의 비용이 모두 동일한 상황 등 특정 조건에서의 최단 경로 문제를 해결하기 위한 목적으로도 효과적으로 사용된다.

1. 탐색 시작 노드를 큐에 삽입하고 방문 처리.

2. 큐에서 노드를 꺼내 해당 노드의 인접 노드 중에서 방문하지 않은 노드를 모두 큐에 삽입하고 방문 처리한다.

3. 2의 과정을 수행할 수 없을 때까지 반복한다.

```python
from collections import deque

def bfs(graph, start, visited):
    queue = deque([start])
    visited[start] = True
    while queue:
        v = queue.popleft()
        print(v, end=' ')
        for i in graph[v]:
            if not visited[i]:
                queue.append(i)
                visited[i] = True
```

---

## [BFS 문제 추천]

- [ ] https://www.acmicpc.net/problem/2667
- [ ] https://www.acmicpc.net/problem/2178
- [ ] https://www.acmicpc.net/problem/14502
- [ ] https://www.acmicpc.net/problem/16236
- [ ] https://www.acmicpc.net/problem/2146
- [ ] https://www.acmicpc.net/problem/2638

## [DFS 문제 추천]

- [ ] https://www.acmicpc.net/problem/2667 (중복이지만 DFS써서 풀어보기.)
- [ ] https://www.acmicpc.net/problem/2468
- [ ] https://www.acmicpc.net/problem/10026
- [ ] https://www.acmicpc.net/problem/1987
- [ ] https://www.acmicpc.net/problem/16437
