"""Breadth-first search (BFS), uninformed.

Frontier: a FIFO queue, so nodes are expanded level by level (shallowest
first). Children are added in ascending node order, which together with the
FIFO order gives the tie-breaking required by the assignment.

Contract: see README.md and AGENTS.md.
"""
from collections import deque


def search(problem):
    """Return (goal, nodes_created, path). goal is None if no goal is reachable."""
    # Each frontier entry is the full path from the origin to that node.
    # Keeping the path lets us rebuild the answer and stop loops on it.
    frontier = deque([[problem.origin]])
    nodes_created = 1  # the origin node

    while frontier:
        path = frontier.popleft()  # oldest entry first (FIFO)
        node = path[-1]

        # Goal test when the node is removed from the frontier (expanded).
        if problem.is_goal(node):
            return node, nodes_created, path

        # neighbors() is already sorted ascending, so smaller nodes go in first.
        for child, _cost in problem.neighbors(node):
            # Tree search: only skip a child already on this path (no cycles).
            if child in path:
                continue
            frontier.append(path + [child])
            nodes_created += 1

    return None, nodes_created, []
