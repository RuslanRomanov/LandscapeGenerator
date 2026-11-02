"""Раскраска ландшафта по высоте: вода, пляж, трава, скалы, снег."""

import numpy as np

# Цвета зон (RGB)
DEEP_WATER = (15, 50, 120)
SHALLOW_WATER = (50, 120, 190)
SAND = (225, 205, 140)
GRASS_LOW = (70, 150, 60)
GRASS_HIGH = (40, 110, 45)
ROCK = (120, 110, 100)
ROCK_HIGH = (150, 145, 140)
SNOW = (245, 247, 250)


def height_colors(heightmap, water_level=0.3):
    """Возвращает цвета узлов карты высот, массив (R, C, 3) uint8.

    heightmap   - значения в [0, 1];
    water_level - уровень воды в [0, 1): всё ниже - вода, цвет зависит от глубины.
    Остальной диапазон делится на зоны: пляж, трава, скалы, снег. Между зонами
    цвета плавно смешиваются.
    """
    if not (0.0 <= water_level < 1.0):
        raise ValueError("water_level должен быть в диапазоне [0, 1)")
    h = np.asarray(heightmap, dtype=np.float64)
    wl = water_level
    span = 1.0 - wl

    # Опорные точки градиента над водой (доли оставшегося диапазона)
    stops = [
        (0.00, SAND),
        (0.06, SAND),
        (0.12, GRASS_LOW),
        (0.45, GRASS_HIGH),
        (0.62, ROCK),
        (0.80, ROCK_HIGH),
        (0.90, SNOW),
        (1.00, SNOW),
    ]
    pos = np.array([wl + s[0] * span for s in stops])
    chan = np.array([s[1] for s in stops], dtype=np.float64)  # (K, 3)

    land = np.stack([np.interp(h, pos, chan[:, c]) for c in range(3)], axis=-1)

    # Вода: от мелкой (у берега) к глубокой (на дне)
    if wl > 0:
        depth = np.clip((wl - h) / wl, 0.0, 1.0)[..., None]
        deep = np.array(DEEP_WATER, dtype=np.float64)
        shallow = np.array(SHALLOW_WATER, dtype=np.float64)
        water = shallow + (deep - shallow) * np.sqrt(depth)
        colors = np.where((h < wl)[..., None], water, land)
    else:
        colors = land
    return np.clip(np.rint(colors), 0, 255).astype(np.uint8)


def apply_lighting(colors, normals, light_dir=(-0.5, 0.8, -0.4), ambient=0.35):
    """Умножает цвета на освещённость (закон Ламберта), цвета и нормали (..., 3)."""
    light = np.asarray(light_dir, dtype=np.float64)
    light = light / np.linalg.norm(light)
    diffuse = np.clip(np.asarray(normals, dtype=np.float64) @ light, 0.0, 1.0)
    k = ambient + (1.0 - ambient) * diffuse
    out = np.asarray(colors, dtype=np.float64) * k[..., None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
