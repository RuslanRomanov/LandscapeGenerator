"""Полный конвейер генерации: шум -> эрозия -> раскраска -> меш."""

import time
from dataclasses import dataclass, field, replace

import numpy as np

from .noise import generate_heightmap
from .erosion import hydraulic_erosion
from .colors import height_colors, apply_lighting
from .mesh import build_mesh, Mesh

WORLD_SIZE = 100.0   # ширина ландшафта в мировых единицах


@dataclass
class TerrainParams:
    """Параметры генерации ландшафта."""

    resolution: int = 128
    seed: int = 0
    scale: float = 3.0
    octaves: int = 5
    persistence: float = 0.5
    lacunarity: float = 2.0
    erosion_iterations: int = 100
    water_level: float = 0.3
    height_scale: float = 30.0

    def with_seed(self, seed):
        return replace(self, seed=seed)


@dataclass
class Landscape:
    """Результат генерации."""

    params: TerrainParams
    heightmap: np.ndarray        # карта высот после эрозии, [0, 1]
    mesh: Mesh
    timings: dict = field(default_factory=dict)


def build_landscape(params):
    """Генерирует ландшафт по параметрам и возвращает Landscape."""
    timings = {}

    t = time.perf_counter()
    hm = generate_heightmap(params.resolution, params.scale, params.seed,
                            params.octaves, params.persistence, params.lacunarity)
    timings["noise"] = time.perf_counter() - t

    t = time.perf_counter()
    if params.erosion_iterations > 0:
        # relief согласован с пропорциями меша: высота / ширина
        relief = params.height_scale / WORLD_SIZE
        hm = hydraulic_erosion(hm, params.erosion_iterations, relief=relief,
                               seed=params.seed)
    timings["erosion"] = time.perf_counter() - t

    t = time.perf_counter()
    # Поверхность воды плоская: всё, что ниже уровня воды, поднимается до него
    shown = np.maximum(hm, params.water_level)
    cell = WORLD_SIZE / (params.resolution - 1)
    mesh = build_mesh(shown, cell_size=cell, height_scale=params.height_scale)
    colors = height_colors(hm, params.water_level)
    # Вода освещается как плоскость, чтобы не показывать рельеф дна
    mesh.albedo = colors.reshape(-1, 3).copy()
    mesh.colors = apply_lighting(colors.reshape(-1, 3), mesh.normals)
    timings["mesh"] = time.perf_counter() - t

    timings["total"] = sum(timings.values())
    return Landscape(params=params, heightmap=hm, mesh=mesh, timings=timings)
