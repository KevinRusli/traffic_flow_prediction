"""Custom method 2 (CUS2), informed: A* search that minimises the number of moves.

Every edge counts as one move, whatever its cost. The frontier is a priority
queue ordered by f(n) = g(n) + h(n), where
    g(n) = moves taken from the origin to n (path length - 1)
    h(n) = 0 if n is a destination, otherwise 1 (any non-goal needs at least
           one more move, so h never overestimates and A* returns a path with
           the fewest moves)
The path is fewest-moves, not lowest-cost.

Contract: see README.md and AGENTS.md.
"""
import heapq


def moves_heuristic(problem, node):
    """Lower bound on the moves still needed: 0 at a goal, otherwise 1."""
    return 0 if problem.is_goal(node) else 1


def search(problem):
    """Return (goal, nodes_created, path). goal is None if no goal is reachable."""
    origin = problem.origin
    nodes_created = 1  # the origin counts as 1

    # Heap entries: (f, node, counter, path). node breaks ties between equal f
    # values (smaller node first), counter breaks ties between equal (f, node)
    # pairs by insertion order (the assignment's tie-breaking rule).
    counter = 0
    frontier = [(moves_heuristic(problem, origin), origin, counter, [origin])]

    while frontier:
        _f, node, _c, path = heapq.heappop(frontier)

        # Goal test on expansion, not on generation.
        if problem.is_goal(node):
            return node, nodes_created, path

        for neighbor, _cost in problem.neighbors(node):
            if neighbor in path:
                continue  # avoid cycling back onto the current path
            child_path = path + [neighbor]
            moves = len(child_path) - 1  # g(child): one move per edge
            nodes_created += 1
            counter += 1
            f = moves + moves_heuristic(problem, neighbor)
            heapq.heappush(frontier, (f, neighbor, counter, child_path))

    return None, nodes_created, []
