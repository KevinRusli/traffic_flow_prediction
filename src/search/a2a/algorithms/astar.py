"""A* search (AS), informed: orders the frontier by f(n) = g(n) + h(n).

g(n) is the cost of the path from the origin to n, and h(n) is the
straight-line distance from n to the nearest destination. The frontier is a
priority queue (heap) of (f, node, counter, g, path): equal f is broken by the
smaller node number, then by the order nodes were added (counter).

Contract: see README.md and AGENTS.md.
"""
import heapq


def search(problem):
    """Return (goal, nodes_created, path). goal is None if no goal is reachable."""
    counter = 0  # increases on every push, used for chronological tie-breaking
    origin = problem.origin
    frontier = [(problem.h(origin), origin, counter, 0, [origin])]
    nodes_created = 1  # the origin node

    while frontier:
        # Lowest f first; ties go to the smaller node, then the earlier push.
        _f, node, _order, g, path = heapq.heappop(frontier)

        # Goal test when the node is removed from the frontier (expanded).
        if problem.is_goal(node):
            return node, nodes_created, path

        for child, cost in problem.neighbors(node):
            # Tree search: only skip a child already on this path (no cycles).
            if child in path:
                continue
            child_g = g + cost
            # Round f so float noise (e.g. 0.1 + 0.2 = 0.30000000000000004)
            # cannot break a tie that is really equal.
            child_f = round(child_g + problem.h(child), 9)
            counter += 1
            heapq.heappush(frontier, (child_f, child, counter, child_g, path + [child]))
            nodes_created += 1

    return None, nodes_created, []