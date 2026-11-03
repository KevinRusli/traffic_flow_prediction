"""Research method ALL: visit every destination with the cheapest total path.

The robot has to reach all destinations in some order, and the cheapest way to
do that is to chain the cheapest route between each pair of consecutive stops
(origin -> first goal -> second goal -> ...).

  1. Run Dijkstra from the origin and from every goal.
  2. Try every order of the goals and add up the cost of each leg.
  3. Take the order with the smallest total and join its legs into one path.

Differences from the six main methods:
  - Edges are directed, so A -> B and B -> A can cost different amounts and the
    order of the goals matters.
  - Dijkstra here keeps the cheapest cost of every node, so it is not a tree
    search. The final path can pass through the same node twice, for example
    when the robot has to turn around after reaching a goal.
  - Trying every order grows as k! for k goals, which is fine for the small
    number of goals in the tests.

search() returns the last goal visited. nodes_created is the sum over all the
Dijkstra runs.
"""
import heapq
from itertools import permutations


def shortest_paths(problem, source):
    """Dijkstra from source. Returns (cost, parent, created).

    cost[n]    cheapest cost from source to n
    parent[n]  node before n on that cheapest route
    created    nodes created (source = 1, plus 1 per push)
    Ties: smaller node number first, then earlier push, as in the other methods.
    """
    cost = {source: 0}
    parent = {}
    counter = 0
    created = 1
    heap = [(0, source, counter)]
    while heap:
        so_far, node, _ = heapq.heappop(heap)
        if so_far > cost[node]:
            continue                # stale entry, a cheaper route was found later
        for neighbour, step in problem.neighbors(node):
            new_cost = so_far + step
            if neighbour not in cost or new_cost < cost[neighbour]:
                cost[neighbour] = new_cost
                parent[neighbour] = node
                counter += 1
                created += 1
                heapq.heappush(heap, (new_cost, neighbour, counter))
    return cost, parent, created


def route(parent, source, target):
    """Rebuild the cheapest route source -> target from the parent dict."""
    nodes = [target]
    while nodes[-1] != source:
        nodes.append(parent[nodes[-1]])
    return nodes[::-1]              # collected from the end, so reverse it


def best_order(problem, runs):
    """Try every order of the goals and return (cheapest total cost, order).

    runs[n] is the shortest_paths result from node n. Returns (None, None) if
    some goal cannot be reached.
    """
    goals = sorted(set(problem.destinations))
    best_total, best = None, None
    for order in permutations(goals):
        total, here, possible = 0, problem.origin, True
        for goal in order:
            leg = runs[here][0].get(goal)       # None if goal is unreachable from here
            if leg is None:
                possible = False
                break
            total += leg
            here = goal
        if possible and (best_total is None or total < best_total):
            best_total, best = total, order
    return best_total, best


def search(problem):
    """Return (last goal, nodes_created, path through every goal)."""
    goals = sorted(set(problem.destinations))
    # one Dijkstra run from the origin and from each goal
    runs = {n: shortest_paths(problem, n) for n in [problem.origin] + goals}
    nodes_created = sum(created for _, _, created in runs.values())

    _, order = best_order(problem, runs)
    if order is None:
        return None, nodes_created, []

    path, here = [problem.origin], problem.origin
    for goal in order:
        leg = route(runs[here][1], here, goal)
        path += leg[1:]             # the leg starts on the node path already ends on
        here = goal
    return order[-1], nodes_created, path
