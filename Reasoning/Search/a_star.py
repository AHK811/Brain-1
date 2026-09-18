from __future__ import annotations
import heapq
def a_star(start, goal_test, neighbors, heuristic, cost=None):
    cost = cost or (lambda a,b: 1.0)
    open_h = [(heuristic(start), 0.0, start, None)]
    came = {}; gscore = {start: 0.0}; closed=set()
    while open_h:
        f,g,node,parent = heapq.heappop(open_h)
        if node in closed: continue
        came[node]=parent
        if goal_test(node):
            path=[node]
            while came.get(path[-1]) is not None: path.append(came[path[-1]])
            path.reverse(); return path
        closed.add(node)
        for nb in neighbors(node):
            ng=g+cost(node,nb)
            if nb in gscore and ng>=gscore[nb]: continue
            gscore[nb]=ng
            heapq.heappush(open_h,(ng+heuristic(nb),ng,nb,node))
    return None
