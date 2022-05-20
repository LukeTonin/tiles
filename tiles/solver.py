from __future__ import annotations
import functools
from dataclasses import dataclass, field
import heapq
import hashlib

import numpy as np

from tiles.game import get_valid_next_grids


def get_grid_path_from_hash(grid_hash: str, hash_to_parent_hash: Dict[str, str], hash_to_grid: Dict[str, np.ndarray]):
    """"""
    grid_path = [hash_to_grid[grid_hash]]

    while True:
        if grid_hash not in hash_to_parent_hash:
            return list(reversed(grid_path))

        parent_hash = hash_to_parent_hash[grid_hash]
        grid_path.append(hash_to_grid[parent_hash])

        grid_hash = parent_hash


def get_distance_to_position(grid: np.ndarray, target_block_value: int, position_x: int = None, position_y: int = None) -> int:

    grid = grid == target_block_value
    x, y = np.unravel_index(np.argmax(grid), grid.shape)

    distance = 0
    if position_x:
        distance += abs(position_x - x)
    if position_y:
        distance += abs(position_y - y) 
    
    return distance


@dataclass(order=True)
class GridWithCost:
    f_cost: float
    grid_hash: str = field(compare=False)


class Solver:
    def __init__(
        self,
        start_grid: np.ndarray,
        end_grid: np.ndarray,
        array_hash_function: Callable = lambda x: hashlib.sha256(x).hexdigest()[:15],
    ):
        self.array_hash_function = array_hash_function
        self.target_block_value = np.unique(end_grid).max()
        self.end_grid_mask = end_grid != 0
        # Get the top left coordinates of target block in the final position.
        self.end_x, self.end_y = np.unravel_index(np.argmax(self.end_grid_mask), self.end_grid_mask.shape)

        self.heuristic_function = functools.partial(
            get_distance_to_position,
            position_x=self.end_x,
            position_y=None,
            target_block_value=self.target_block_value,
        )

        start_grid_hash = array_hash_function(start_grid)

        self.hash_to_grid = {start_grid_hash: start_grid}
        self.hash_to_g_cost = {start_grid_hash: 0}
        self.hash_to_h_cost = {start_grid_hash: self.heuristic_function(grid=start_grid)}

        self.hash_to_parent_hash = {}

        start_grid_with_cost = GridWithCost(f_cost=self._get_total_cost(grid_hash=start_grid_hash), grid_hash=start_grid_hash)
        self.queue = [start_grid_with_cost]
        self.hash_to_grid_with_cost = {start_grid_hash: start_grid_with_cost}
    
    def _pop_from_queue(self) -> GridWithCost:

        while self.queue:
            grid_with_cost = heapq.heappop(self.queue)

            if grid_with_cost.grid_hash is not None:
                del self.hash_to_grid_with_cost[grid_with_cost.grid_hash]
                return grid_with_cost
        
        raise Exception('Could not solve grid.')
    
    def _remove_from_queue(self, grid_hash: str):
        """"""
        grid_with_cost = self.hash_to_grid_with_cost.pop(grid_hash)
        grid_with_cost.grid_hash = None


    def _get_total_cost(self, grid_hash: str) -> float:
        """""" 
        return self.hash_to_g_cost[grid_hash] + self.hash_to_h_cost[grid_hash]

    def _update_cost(self, grid_hash: str) -> None:
        """"""
        parent_hash = self.hash_to_parent_hash[grid_hash]
        self.hash_to_g_cost[self.hash_to_g_cost[parent_hash] + 1]

        if grid_hash not in self.hash_to_h_cost:
            grid = self.hash_to_grid[grid_hash]
            self.hash_to_h_cost[grid_hash] = self.heuristic_function(grid=grid)

    def solve(self, log_frequency: int = 1000) -> str:
        """"""

        while self.queue:
            grid_with_cost = self._pop_from_queue()
            grid_hash = grid_with_cost.grid_hash
            grid = self.hash_to_grid[grid_hash]

            # if self.hash_to_h_cost[grid_hash] == 0:
            if get_distance_to_position(grid=grid, position_x=self.end_x, position_y=self.end_y, target_block_value=self.target_block_value) == 0:
                return get_grid_path_from_hash(
                    grid_hash=grid_hash, hash_to_parent_hash=self.hash_to_parent_hash, hash_to_grid=self.hash_to_grid
                )
            
            neighbor_grids = get_valid_next_grids(grid=grid)

            if len(self.hash_to_grid) % log_frequency == 0 and len(self.hash_to_grid) != 0:
                grid_path = get_grid_path_from_hash(
                    grid_hash=grid_hash, hash_to_parent_hash=self.hash_to_parent_hash, hash_to_grid=self.hash_to_grid
                )
                print(f'Length of grid path: {len(grid_path)}', end = ' ')
                print(f'Num grids analysed: {len(self.hash_to_grid)}', end = ' ')
                print(f'Length of queue: {len(self.queue)}', end = ' ')
                print(f'g_cost: {self.hash_to_g_cost[grid_hash]}', end = ' ')
                print(f'h_cost: {self.hash_to_h_cost[grid_hash]}', end = ' ')
                print(f'min_h_cost: {min(self.hash_to_h_cost.values())}', end = ' ')

                print()

            for neighbor_grid in neighbor_grids:

                tentative_g_cost = self.hash_to_g_cost[grid_hash] + 1
                neighbor_hash = self.array_hash_function(neighbor_grid)
                
                if neighbor_hash not in self.hash_to_grid or tentative_g_cost < self.hash_to_g_cost[neighbor_hash]:
                    self.hash_to_grid[neighbor_hash] = neighbor_grid
                    self.hash_to_parent_hash[neighbor_hash] = grid_hash
                    self.hash_to_g_cost[neighbor_hash] = tentative_g_cost
                    h_cost = self.heuristic_function(grid=neighbor_grid)
                    self.hash_to_h_cost[neighbor_hash] = h_cost
                    neighbor_grid_with_cost = GridWithCost(f_cost=tentative_g_cost + h_cost * 20, grid_hash=neighbor_hash)

                    if neighbor_hash in self.hash_to_grid_with_cost:
                        self._remove_from_queue(grid_hash=neighbor_hash)

                    heapq.heappush(self.queue, neighbor_grid_with_cost)
                    self.hash_to_grid_with_cost[neighbor_hash] = neighbor_grid_with_cost
                    

# function A_Star(start, goal, h)
#     // The set of discovered nodes that may need to be (re-)expanded.
#     // Initially, only the start node is known.
#     // This is usually implemented as a min-heap or priority queue rather than a hash-set.
#     openSet := {start}

#     // For node n, cameFrom[n] is the node immediately preceding it on the cheapest path from start
#     // to n currently known.
#     cameFrom := an empty map

#     // For node n, gScore[n] is the cost of the cheapest path from start to n currently known.
#     gScore := map with default value of Infinity
#     gScore[start] := 0

#     // For node n, fScore[n] := gScore[n] + h(n). fScore[n] represents our current best guess as to
#     // how short a path from start to finish can be if it goes through n.
#     fScore := map with default value of Infinity
#     fScore[start] := h(start)

#     while openSet is not empty
#         // This operation can occur in O(Log(N)) time if openSet is a min-heap or a priority queue
#         current := the node in openSet having the lowest fScore[] value
#         if current = goal
#             return reconstruct_path(cameFrom, current)

#         openSet.Remove(current)
#         for each neighbor of current
#             // d(current,neighbor) is the weight of the edge from current to neighbor
#             // tentative_gScore is the distance from start to the neighbor through current
#             tentative_gScore := gScore[current] + d(current, neighbor)
#             if tentative_gScore < gScore[neighbor]
#                 // This path to neighbor is better than any previous one. Record it!
#                 cameFrom[neighbor] := current
#                 gScore[neighbor] := tentative_gScore
#                 fScore[neighbor] := tentative_gScore + h(neighbor)
#                 if neighbor not in openSet
#                     openSet.add(neighbor)

#     // Open set is empty but goal was never reached
#     return failure
