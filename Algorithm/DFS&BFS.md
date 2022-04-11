# DFS(Depth-First Search: 깊이 우선 탐색)

깊은 부분을 우선적으로 탐색하는 알고리즘. 스택이나 재귀함수를 이용해 구현.

1. 탐색 시작 노드를 스택에 삽입한 후 방문 처리를 한다.

2. 스택의 최상단 노드와 인접한 노드 중에 방문하지 않은 노드가 있으면 해당 노드를 스택에 넣고 방문 처리를 한다. 방문하지 않은 노드가 없다면(인접 노드를 모두 방문 했다면) 스택의 최상단 노드에서 꺼낸다.

3. 2의 과정을 수행할 수 없을 때까지 반복한다.

```python
# 재귀
def dfs_recur(graph, v, visited):
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

dfs_recur(graph, 1, visited)
# 1 2 7 6 8 3 4 5

# 스택
def dfs_iter(graph, start):
    visited = []
    stack = []

    stack.append(start)

    while stack:
        v = stack.pop()
        if v not in visited:
            visited.append(v)
            stack.extend(graph[v])

    return visited
```

### DFS의 장점

현 경로상의 노드를 기억하기 때문에 적은 메모리를 사용한다.  
찾으려는 노드가 깊은 단계에 있는 경우 BFS보다 빠르게 찾을 수 있다.

### DFS의 단점

해가 없는 경로를 탐색할 경우 단계가 끝날 때까지 탐색한다. 효율성을 높이기 위해 미리 지정한 임의 깊이까지만 탐색하고 해를 발견하지 못하면 빠져나와 다른 경로를 탐색하는 방법을 사용한다.  
DFS를 통해 얻어진 해가 최단 경로라는 보장이 없다. -> 해에 도착하면 탐색을 종료하기 때문.

# BFS(Breadth-First Search: 너비 우선 탐색)

가까운 노드부터 우선 탐색하는 알고리즘. 큐 자료구조를 이용.  
각 간선의 비용이 모두 동일한 상황 등 특정 조건에서의 최단 경로 문제를 해결하기 위한 목적으로도 효과적으로 사용된다.

1. 탐색 시작 노드를 큐에 삽입하고 방문 처리.

2. 큐에서 노드를 꺼내 해당 노드의 인접 노드 중에서 방문하지 않은 노드를 모두 큐에 삽입하고 방문 처리한다.

3. 2의 과정을 수행할 수 없을 때까지 반복한다.

```python
from collections import deque

def bfs(graph, start):
    visited = []
    queue = deque([start])

    while queue:
        v = queue.popleft()
        if v not in visited:
            visited.append(v)
            queue.extend(graph[v])
    return visited
```

### BFS의 장점

답이 되는 경로가 여러 개인 경우에도 최단 경로임을 보장한다.  
최단 경로가 존재하면 깊이가 무한정 깊어진다고 해도 답을 찾을 수 있다.

### BFS의 단점

경로가 매우 길 경우 탐색 가지가 급격히 증가해 많은 기억 공간을 필요로 하게 된다.  
해가 존재하지 않는다면 유한 그래프의 경우 모든 그래프를 탐색 후 실패로 끝난다. 무한 그래프의 경우에는 해를 찾지도 못하고 끝내지도 못한다.

## DFS vs BFS 차이

DFS는 스택(또는 재귀), BFS는 큐로 구현한다. (BFS는 재귀적으로 동작하지 않는다.)

**문제 풀이 시**  
최단 거리 문제: BFS 사용.  
이동할 때마다 가중치가 붙어서 이동하거나 이동 과정에서 여러 제약이 있을 경우: DFS 사용.

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

## [문제 추천]

- [ ] https://www.acmicpc.net/problem/1260
- [ ] https://www.acmicpc.net/problem/1697
- [ ] https://www.acmicpc.net/problem/11724
- [ ] https://www.acmicpc.net/problem/6603
- [ ] https://www.acmicpc.net/problem/7576
- [ ] https://www.acmicpc.net/problem/7562
