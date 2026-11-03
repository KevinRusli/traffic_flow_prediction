"""Depth-first search (DFS), uninformed.

Frontier is a stack (LIFO): the most recently added node is expanded next.
Children are pushed in descending order, so the smallest-numbered child ends
up on top of the stack and is expanded first (the assignment's tie-breaking
rule).

Contract: see README.md and AGENTS.md.
"""


def search(problem):
    """Return (goal, nodes_created, path). goal is None if no goal is reachable."""
    origin = problem.origin
    nodes_created = 1  # the origin counts as 1

    # Each stack entry is the path so far (list of nodes) from origin to that
    # node. Using the path itself (instead of a separate visited set) lets us
    # apply the only loop check this assignment allows: never add a node that
    # is already on the current path.
    stack = [[origin]]

    while stack:
        path = stack.pop()
        node = path[-1]

        if problem.is_goal(node):
            return node, nodes_created, path

        # Push children in reverse ascending order so that, since the stack
        # is LIFO, they get popped in ascending order (smallest node first).
        for neighbor, _cost in reversed(problem.neighbors(node)):
            if neighbor in path:
                continue  # avoid cycling back onto the current path
            nodes_created += 1
            stack.append(path + [neighbor])

    return None, nodes_created, []