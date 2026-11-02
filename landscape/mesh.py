"""Построение 3D-меша из карты высот."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Mesh:
    """Регулярный меш-сетка.

    vertices - массив (rows * cols, 3) float32, координаты (x, y, z); y - вверх.
    faces    - массив (M, 3) int32, треугольники (индексы вершин).
    rows     - число строк сетки (вдоль оси z).
    cols     - число столбцов сетки (вдоль оси x).
    normals  - необязательный массив (rows * cols, 3) float32, нормали вершин.
    colors   - необязательный массив (rows * cols, 3) uint8, цвета вершин
               (с учётом освещения, если оно применено) - для экрана Pygame.
    albedo   - необязательный массив (rows * cols, 3) uint8, "чистые" цвета
               вершин без освещения - для экспорта (освещение в Blender и
               других программах считается заново).
    """

    vertices: np.ndarray
    faces: np.ndarray
    rows: int
    cols: int
    normals: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    albedo: Optional[np.ndarray] = None

    def grid(self):
        """Вершины в виде массива (rows, cols, 3)."""
        return self.vertices.reshape(self.rows, self.cols, 3)

    def color_grid(self):
        """Цвета вершин в виде массива (rows, cols, 3) или None."""
        if self.colors is None:
            return None
        return self.colors.reshape(self.rows, self.cols, 3)


def compute_normals(vertices, rows, cols):
    """Нормали вершин регулярной сетки через конечные разности, (rows*cols, 3)."""
    g = np.asarray(vertices, dtype=np.float64).reshape(rows, cols, 3)
    dx = np.gradient(g, axis=1)   # вдоль столбцов (x)
    dz = np.gradient(g, axis=0)   # вдоль строк (z)
    n = np.cross(dz, dx)
    # Ориентация "вверх": для плоской сетки результат (0, +1, 0)
    n[n[..., 1] < 0] *= -1.0
    length = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(length, 1e-12)
    return n.reshape(-1, 3).astype(np.float32)


def build_mesh(heightmap, cell_size=1.0, height_scale=20.0):
    """Строит меш из карты высот.

    heightmap    - массив (rows, cols) со значениями высоты (обычно 0..1).
    cell_size    - расстояние между соседними узлами по x и z.
    height_scale - множитель высоты (координата y).

    Меш центрирован по x и z относительно начала координат. Нормали
    вычисляются сразу, цвета можно присвоить позже (mesh.colors = ...).
    """
    hm = np.asarray(heightmap, dtype=np.float32)
    if hm.ndim != 2 or hm.shape[0] < 2 or hm.shape[1] < 2:
        raise ValueError("Карта высот должна быть двумерной, минимум 2 x 2")

    rows, cols = hm.shape
    xs = (np.arange(cols, dtype=np.float32) - (cols - 1) / 2.0) * cell_size
    zs = (np.arange(rows, dtype=np.float32) - (rows - 1) / 2.0) * cell_size
    gx, gz = np.meshgrid(xs, zs)  # оба (rows, cols)

    vertices = np.stack([gx, hm * height_scale, gz], axis=-1).reshape(-1, 3)

    # Два треугольника на каждую ячейку сетки
    idx = np.arange(rows * cols, dtype=np.int32).reshape(rows, cols)
    a = idx[:-1, :-1].ravel()
    b = idx[:-1, 1:].ravel()
    c = idx[1:, :-1].ravel()
    d = idx[1:, 1:].ravel()
    faces = np.concatenate(
        [np.stack([a, c, b], axis=1), np.stack([b, c, d], axis=1)], axis=0
    )

    normals = compute_normals(vertices, rows, cols)
    return Mesh(vertices=vertices, faces=faces, rows=rows, cols=cols,
                normals=normals)
