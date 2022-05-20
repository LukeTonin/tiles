from __future__ import annotations
import enum

import numpy as np


class Direction(enum.Enum):
    UP = enum.auto()
    DOWN = enum.auto()
    LEFT = enum.auto()
    RIGHT = enum.auto()

class TilesEnv:
    pass


def get_blocks_in_direction(grid: np.ndarray, block_value: int, direction: Direction) -> np.ndarray:
    """"""

    block_grid = grid == block_value

    if direction == Direction.UP:
        blocks = grid[:-1, :][block_grid[1:, :]]

    elif direction == Direction.DOWN:
        blocks = grid[1:, :][block_grid[:-1, :]]

    elif direction == Direction.LEFT:
        blocks = grid[:, :-1][block_grid[:, 1:]]

    elif direction == Direction.RIGHT:
        blocks = grid[:, 1:][block_grid[:, :-1]]

    else:
        ValueError("direction must be an instance of Direction")

    blocks = blocks[blocks != block_value]

    return blocks


def calculate_is_valid_move(grid: np.ndarray, block_value: int, direction: Direction) -> bool:
    """"""

    blocks = get_blocks_in_direction(grid=grid, block_value=block_value, direction=direction)
    is_valid_move = blocks.size != 0 and blocks.max() == 0
    return is_valid_move


def move_block(grid: np.ndarray, block_value: int, direction: Direction) -> np.ndarray:
    """"""
    new_grid = grid.copy()
    block_grid = grid == block_value
    new_grid[block_grid] = 0

    if direction == Direction.UP:
        new_grid[:-1, :][block_grid[1:, :]] = block_value

    elif direction == direction.DOWN:
        new_grid[1:, :][block_grid[:-1, :]] = block_value

    elif direction == Direction.LEFT:
        new_grid[:, :-1][block_grid[:, 1:]] = block_value

    elif direction == Direction.RIGHT:
        new_grid[:, 1:][block_grid[:, :-1]] = block_value

    else:
        ValueError("direction must be an instance of Direction")

    return new_grid


def get_valid_next_grids(grid: np.ndarray) -> List[np.ndarray]:
    block_values = np.unique(grid)
    block_values = block_values[block_values != 0]

    possible_new_grids: List[np.ndarray] = []

    for block_value in block_values:

        for direction in Direction:
            is_move_valid = calculate_is_valid_move(grid=grid, block_value=block_value, direction=direction)

            if is_move_valid:
                new_grid = move_block(grid=grid, block_value=block_value, direction=direction)
                possible_new_grids.append(new_grid)

    return possible_new_grids


def get_valid_next_moves(grid: np.ndarray) -> List[Tuple[int, Direction]]:
    block_values = np.unique(grid)
    block_values = block_values[block_values != 0]

    possible_new_moves: List[Tuple[int, Direction]] = []

    for block_value in block_values:

        for direction in Direction:
            is_move_valid = calculate_is_valid_move(grid=grid, block_value=block_value, direction=direction)

            if is_move_valid:
                new_grid = move_block(grid=grid, block_value=block_value, direction=direction)
                possible_new_moves.append((block_value, direction))

    return possible_new_moves


def create_grids_from_moves(grid: np.ndarray, moves: List[Tuple[int, Direction]]) -> List[np.ndarry]:
    """"""

    grids: List[np.ndarray] = [grid]

    for block_value, direction in moves:
        grid = move_block(grid=grid, block_value=block_value, direction=direction)
        grids.append(grid)

    return grids
