"""Greedy best-first search (GBFS), informed.

Frontier is a priority queue ordered only by the heuristic h(n): the
estimated straight-line distance from a node to the nearest destination.
The cost already paid to reach a node is ignored, which is what makes this
"greedy" rather than A*.

Contract: see README.md and AGENTS.md.
"""
import heapq


def search(problem):
    """Return (goal, nodes_created, path). goal is None if no goal is reachable."""
    origin = problem.origin
    nodes_created = 1  # the origin counts as 1

    # Heap entries: (h(node), node, counter, path). node breaks ties between
    # equal h values (smaller node first), counter breaks ties between equal
    # (h, node) pairs by insertion order (the assignment's tie-breaking rule).
    counter = 0
    frontier = [(problem.h(origin), origin, counter, [origin])]

    while frontier:
        _h, node, _c, path = heapq.heappop(frontier)

        if problem.is_goal(node):
            return node, nodes_created, path

        for neighbor, _cost in problem.neighbors(node):
            if neighbor in path:
                continue  # avoid cycling back onto the current path
            nodes_created += 1
            counter += 1
            heapq.heappush(frontier, (problem.h(neighbor), neighbor, counter, path + [neighbor]))

    return None, nodes_created, []
