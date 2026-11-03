"""CUS1: bidirectional breadth-first search (uninformed).

A forward search starts at the origin and a backward search starts at every
destination and follows the edges in reverse. They take turns, forward first.
On its turn a side takes the oldest path in its queue (FIFO) and extends it by
every neighbour in ascending node order, skipping neighbours already on that path.

The search stops as soon as a node created by one side was already reached by
the other, and joins the two paths at that node. This meeting replaces the usual
goal test when a node leaves the frontier.

Only edges are used (no coordinates or costs), so it is uninformed. It returns a
path with the fewest edges, which is not always the cheapest one. nodes_created
starts at 1 + the number of destinations, then adds 1 for each child created.
"""
from collections import deque


def reverse_graph(problem):
    """Return the graph with every edge reversed (a -> b becomes b -> a).

    The cost stays with the edge. This reads problem.graph directly because it
    needs every edge, not only the neighbours of one node.
    """
    reversed_graph = {node: [] for node in problem.graph}

    for a in problem.graph:
        for b, cost in problem.graph[a]:
            reversed_graph[b].append((a, cost))

    for node in reversed_graph:
        reversed_graph[node].sort()

    return reversed_graph


def join_paths(forward_path, backward_path):
    """Join two paths that end on the same node. [2, 1] + [4, 1] -> [2, 1, 4]."""
    # the backward path runs goal -> meeting node, so reverse it and drop the
    # meeting node, which forward_path already ends with
    return forward_path + backward_path[::-1][1:]


def expand_one(queue, graph, reached, other_reached):
    """Take the oldest path off the queue and extend it by each neighbour.

    Returns (children_created, meeting_node). meeting_node is None unless a new
    child was already reached by the other side. It stops at the first meeting,
    so the remaining neighbours of that node are not created.
    """
    path = queue.popleft()
    node = path[-1]                 # the path ends on the node we expand
    created = 0

    for neighbour, _ in graph[node]:
        if neighbour in path:       # skip loops
            continue
        new_path = path + [neighbour]
        queue.append(new_path)
        created += 1
        if neighbour not in reached:
            reached[neighbour] = new_path       # keep only the first path to a node

        if neighbour in other_reached:          # the two searches met
            return created, neighbour

    return created, None


def result_at(meeting, forward_reached, backward_reached, nodes_created):
    """Build the (goal, nodes_created, path) answer once the two sides meet."""
    forward_path = forward_reached[meeting]
    backward_path = backward_reached[meeting]
    goal = backward_path[0]         # a backward path starts on a destination
    return goal, nodes_created, join_paths(forward_path, backward_path)


def search(problem):
    """Return (goal, nodes_created, path). goal is None if no goal is reachable."""
    if problem.is_goal(problem.origin):
        return problem.origin, 1, [problem.origin]

    backward_graph = reverse_graph(problem)
    goals = sorted(problem.destinations)

    # the queues hold whole paths, not single nodes
    forward_queue = deque([[problem.origin]])
    backward_queue = deque([[g] for g in goals])

    # first path found to each node, used to spot the meeting
    forward_reached = {problem.origin: [problem.origin]}
    backward_reached = {g: [g] for g in goals}

    nodes_created = 1 + len(goals)  # the origin and every destination

    while forward_queue or backward_queue:
        # forward turn
        if forward_queue:
            created, meeting = expand_one(forward_queue, problem.graph,
                                          forward_reached, backward_reached)
            nodes_created += created
            if meeting is not None:
                return result_at(meeting, forward_reached, backward_reached, nodes_created)

        # backward turn: same step on the reversed graph, with the roles swapped
        if backward_queue:
            created, meeting = expand_one(backward_queue, backward_graph,
                                          backward_reached, forward_reached)
            nodes_created += created
            if meeting is not None:
                return result_at(meeting, forward_reached, backward_reached, nodes_created)

    return None, nodes_created, []  # both queues are empty: no route
