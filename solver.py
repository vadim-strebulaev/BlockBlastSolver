from __future__ import annotations

import os
from itertools import permutations
from typing import Dict, List, Optional, Sequence, Tuple

from main import (
    DEFAULT_DEVICE_ID,
    capture_all,
    image_to_board_matrix,
    image_to_figures_matrices,
    read_image,
)
from place import place_figure

Board = List[List[int]]
Figure = List[List[int]]


def trim_figure(figure_5x5: Figure) -> Figure:
    """Обрезает пустые строки/столбцы вокруг фигуры."""
    ones = [
        (r, c)
        for r in range(len(figure_5x5))
        for c in range(len(figure_5x5[r]))
        if figure_5x5[r][c] == 1
    ]
    if not ones:
        return [[0]]

    min_r = min(r for r, _ in ones)
    max_r = max(r for r, _ in ones)
    min_c = min(c for _, c in ones)
    max_c = max(c for _, c in ones)

    return [
        [figure_5x5[r][c] for c in range(min_c, max_c + 1)]
        for r in range(min_r, max_r + 1)
    ]


def figure_cells(figure_5x5: Figure) -> List[Tuple[int, int]]:
    """Возвращает координаты заполненных клеток фигуры относительно центра [2][2] массива 5x5."""
    return [
        (r - 2, c - 2)
        for r in range(len(figure_5x5))
        for c in range(len(figure_5x5[r]))
        if figure_5x5[r][c] == 1
    ]


def placement_reason(
    board: Board, cells: Sequence[Tuple[int, int]], row: int, col: int
) -> Tuple[bool, str]:
    """Проверка, можно ли поставить фигуру в (row, col), и причина отказа."""
    for dr, dc in cells:
        rr = row + dr
        cc = col + dc
        if rr < 0 or rr >= 8 or cc < 0 or cc >= 8:
            return False, "выходит за рамки"

    for dr, dc in cells:
        rr = row + dr
        cc = col + dc
        if board[rr][cc] == 1:
            return False, "занято"

    return True, "можно"


def apply_placement(
    board: Board,
    cells: Sequence[Tuple[int, int]],
    row: int,
    col: int,
) -> Tuple[Board, int, int, int]:
    """Ставит фигуру и очищает заполненные ряды/колонки."""
    next_board = [r[:] for r in board]
    for dr, dc in cells:
        next_board[row + dr][col + dc] = 1

    rows_to_clear = [
        r for r in range(8) if all(next_board[r][c] == 1 for c in range(8))
    ]
    cols_to_clear = [
        c for c in range(8) if all(next_board[r][c] == 1 for r in range(8))
    ]

    for r in rows_to_clear:
        for c in range(8):
            next_board[r][c] = 0

    for c in cols_to_clear:
        for r in range(8):
            next_board[r][c] = 0

    return (
        next_board,
        len(rows_to_clear),
        len(cols_to_clear),
        len(rows_to_clear) + len(cols_to_clear),
    )


def print_board(board: Board) -> None:
    for row in board:
        print(row)


def scan_figure_positions(
    board: Board, figure_id: int, figure_cells_rel: Sequence[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Полный перебор 64 клеток для одной фигуры (диагностический вывод)."""
    valid_positions: List[Tuple[int, int]] = []
    for row in range(8):
        for col in range(8):
            ok, reason = placement_reason(board, figure_cells_rel, row, col)
            # print(f"фигура {figure_id} клетка {row} {col} - {reason}")
            if ok:
                valid_positions.append((row, col))

    if not valid_positions:
        print(f"фигуру {figure_id} некуда разместить")

    return valid_positions


def find_best_turn_solution(
    board: Board,
    figures: Dict[int, Figure],
) -> Tuple[Optional[List[Tuple[int, int, int, int]]], int, int, int]:
    """Ищет лучшее решение хода с отсечениями.

    Критерий оптимальности:
    1) максимум суммарных очищений рядов/колонок,
    2) при равенстве минимум островков 0 в итоговом поле.

    Возвращает:
    - best_solution (или None, если решений нет),
    - best_cleared,
    - best_islands,
    - solutions_count (сколько полных валидных решений найдено).
    """

    figure_cells_map: Dict[int, List[Tuple[int, int]]] = {
        fig_id: figure_cells(fig) for fig_id, fig in figures.items()
    }

    best_solution: Optional[List[Tuple[int, int, int, int]]] = None
    best_cleared = -1
    best_islands = 10**9
    solutions_count = 0

    max_clear_per_move = 16  # 8 рядов + 8 колонок

    def dfs(
        current_board: Board,
        order: Sequence[int],
        idx: int,
        path: List[Tuple[int, int, int, int]],
        cleared_so_far: int,
    ) -> None:
        nonlocal best_solution, best_cleared, best_islands, solutions_count

        remaining = len(order) - idx
        if cleared_so_far + remaining * max_clear_per_move < best_cleared:
            return

        if idx == len(order):
            solutions_count += 1
            islands = count_zero_islands(current_board)

            if (
                cleared_so_far > best_cleared
                or (cleared_so_far == best_cleared and islands < best_islands)
            ):
                best_cleared = cleared_so_far
                best_islands = islands
                best_solution = path[:]
            return

        fig_id = order[idx]
        cells = figure_cells_map[fig_id]

        candidates: List[Tuple[int, int, int, Board]] = []

        for row in range(8):
            for col in range(8):
                ok, _ = placement_reason(current_board, cells, row, col)
                if not ok:
                    continue

                next_board, _, _, total_cleared = apply_placement(
                    current_board, cells, row, col
                )

                candidates.append((total_cleared, row, col, next_board))

        if not candidates:
            return

        # Сначала пробуем наиболее "выгодные" локально ходы,
        # чтобы быстрее поднять best_cleared и усилить отсечения.
        candidates.sort(key=lambda t: t[0], reverse=True)

        for total_cleared, row, col, next_board in candidates:
                path.append((fig_id, row, col, total_cleared))
                dfs(next_board, order, idx + 1, path, cleared_so_far + total_cleared)
                path.pop()

    for order in permutations([1, 2, 3]):
        dfs(board, order, 0, [], 0)

    return best_solution, best_cleared, best_islands, solutions_count


def count_zero_islands(board: Board) -> int:
    """Считает количество островов (4-связных кластеров) клеток со значением 0."""
    visited = [[False] * 8 for _ in range(8)]
    islands = 0

    for r in range(8):
        for c in range(8):
            if board[r][c] != 0 or visited[r][c]:
                continue

            islands += 1
            stack = [(r, c)]
            visited[r][c] = True

            while stack:
                cr, cc = stack.pop()
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr = cr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < 8
                        and 0 <= nc < 8
                        and not visited[nr][nc]
                        and board[nr][nc] == 0
                    ):
                        visited[nr][nc] = True
                        stack.append((nr, nc))

    return islands


def honest_offsets_from_figure(_figure_5x5: Figure) -> Tuple[int, int]:
    """Возвращает флаги чётности размеров фигуры по bbox:
    чётное -> 1, нечётное -> 0.
    """
    leftmost = 5
    rightmost = -1
    topmost = 5
    bottommost = -1

    for r in range(5):
        for c in range(5):
            if _figure_5x5[r][c] == 1:
                leftmost = min(leftmost, c)
                rightmost = max(rightmost, c)
                topmost = min(topmost, r)
                bottommost = max(bottommost, r)

    # Пустая фигура (защита)
    if rightmost == -1:
        return 0, 0

    width = rightmost - leftmost + 1
    height = bottommost - topmost + 1

    honestX = 1 if (width % 2 == 0) else 0
    honestY = 1 if (height % 2 == 0) else 0

    return honestX, honestY


def main() -> None:
    device_id = os.getenv("BLOCK_DEVICE_ID", DEFAULT_DEVICE_ID)
    fallback_image = "screen.png"
    try:
        out = capture_all(device_id=device_id)
        board: Board = out["board"]
        figures_raw: List[Figure] = out["figures"]
        print(f"Состояние получено с устройства: {device_id}")
    except Exception as e:
        print(
            f"Не удалось снять скрин через ADB ({e}). Беру локальный {fallback_image}"
        )
        img = read_image(fallback_image)
        board = image_to_board_matrix(img)
        figures_raw = image_to_figures_matrices(img)

    figures = {1: figures_raw[0], 2: figures_raw[1], 3: figures_raw[2]}

    print("Текущее поле (8x8):")
    print_board(board)
    print()

    print("Текущие фигуры (5x5):")
    for fig_id in (1, 2, 3):
        print(f"Figure {fig_id}:")
        for row in figures[fig_id]:
            print(row)
    print()

    print("Диагностика: полный перебор каждой фигуры на всех 64 клетках текущего поля")
    figure_cells_map = {fig_id: figure_cells(fig) for fig_id, fig in figures.items()}
    for fig_id in (1, 2, 3):
        scan_figure_positions(board, fig_id, figure_cells_map[fig_id])
    print()

    solution, best_cleared, best_islands, solutions_count = find_best_turn_solution(
        board, figures
    )

    if solution is None:
        print("Решения на ход нет: нельзя разместить все 3 фигуры ни в одном порядке.")
        return

    if solutions_count == 1:
        print("единственное решение:")
    else:
        print(f"найдено решений: {solutions_count}")
        print(
            f"выбрано решение: максимум очищений={best_cleared},"
            f" минимум островков 0={best_islands}"
        )

    for fig_id, row, col, cleared in solution:
        # print(f"фигура {fig_id} клетка {row} {col}")
        if cleared > 0:
            print(f"удаляются {cleared} ряда/колонки")

    print("ход закончен")

    if solutions_count == 1:
        print("\nРешение единственное — выполняю свайпы через place.py:")
    else:
        print(
            "\nРешений несколько — применяю выбранное (максимум очищений, затем минимум островков) через place.py:"
        )

    for fig_id, row, col, _ in solution:
        honest_x, honest_y = honest_offsets_from_figure(figures[fig_id])
        place_figure(
            figure_id=fig_id,
            col=col,
            row=row,
            honestX=honest_x,
            honestY=honest_y,
            device_id=device_id,
        )


if __name__ == "__main__":
    main()
