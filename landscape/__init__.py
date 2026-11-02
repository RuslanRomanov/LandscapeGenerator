"""Процедурный генератор 3D-ландшафта (этап 1).

Модули:
    noise    - шум Перлина и построение карты высот
    mesh     - построение 3D-меша из карты высот
    camera   - камера с фиксированным углом обзора
    renderer - wireframe-визуализация в Pygame
"""

from .noise import perlin_noise_2d, generate_heightmap
from .mesh import Mesh, build_mesh
from .camera import Camera

__all__ = [
    "perlin_noise_2d",
    "generate_heightmap",
    "Mesh",
    "build_mesh",
    "Camera",
]
