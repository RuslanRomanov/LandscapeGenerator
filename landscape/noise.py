"""Шум Перлина (2D), полностью векторизованный через NumPy."""

import numpy as np

# Допустимый диапазон разрешения карты высот (ограничение по производительности)
MIN_RESOLUTION = 8
MAX_RESOLUTION = 256


def _fade(t):
    """Сглаживающая кривая Перлина: 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def perlin_noise_2d(width, height, scale=4.0, seed=0):
    """Возвращает массив (height, width) с шумом Перлина.

    width, height - размер массива в пикселях (узлах сетки высот).
    scale         - число ячеек шума вдоль большей стороны массива
                    (чем больше, тем мельче детали).
    seed          - зерно генератора, одинаковый seed даёт одинаковый результат.

    Значения лежат примерно в диапазоне [-1, 1] (нормализация не делается).
    """
    if width < 1 or height < 1:
        raise ValueError("Размер массива должен быть положительным")
    if scale <= 0:
        raise ValueError("scale должен быть больше нуля")

    rng = np.random.default_rng(seed)

    # Размер ячейки в пикселях одинаков по обеим осям
    cell = max(width, height) / float(scale)
    xs = np.arange(width) / cell    # (W,)
    ys = np.arange(height) / cell   # (H,)

    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    fx = xs - x0                    # дробная часть, (W,)
    fy = ys - y0                    # (H,)

    # Случайные единичные градиенты в узлах решётки
    nodes_x = int(x0.max()) + 2
    nodes_y = int(y0.max()) + 2
    angles = rng.uniform(0.0, 2.0 * np.pi, size=(nodes_y, nodes_x))
    grad_x = np.cos(angles)
    grad_y = np.sin(angles)

    # Индексы узлов: строки - по y, столбцы - по x (через broadcasting)
    ix0 = x0[None, :]
    ix1 = ix0 + 1
    iy0 = y0[:, None]
    iy1 = iy0 + 1
    dx0 = fx[None, :]
    dx1 = dx0 - 1.0
    dy0 = fy[:, None]
    dy1 = dy0 - 1.0

    # Скалярные произведения градиентов с векторами до точки
    n00 = grad_x[iy0, ix0] * dx0 + grad_y[iy0, ix0] * dy0
    n10 = grad_x[iy0, ix1] * dx1 + grad_y[iy0, ix1] * dy0
    n01 = grad_x[iy1, ix0] * dx0 + grad_y[iy1, ix0] * dy1
    n11 = grad_x[iy1, ix1] * dx1 + grad_y[iy1, ix1] * dy1

    # Билинейная интерполяция со сглаживанием
    u = _fade(dx0)
    v = _fade(dy0)
    nx0 = n00 + u * (n10 - n00)
    nx1 = n01 + u * (n11 - n01)
    result = nx0 + v * (nx1 - nx0)

    # Для единичных градиентов в 2D максимум шума равен sqrt(2)/2
    return result / (np.sqrt(2.0) / 2.0)


def fbm_noise_2d(width, height, scale=4.0, octaves=5, persistence=0.5,
                 lacunarity=2.0, seed=0):
    """Фрактальный шум (fBm): сумма октав шума Перлина.

    octaves     - число октав (каждая следующая мельче и слабее);
    persistence - во сколько раз уменьшается амплитуда на каждой октаве;
    lacunarity  - во сколько раз растёт частота на каждой октаве.

    Результат делится на сумму амплитуд, поэтому лежит примерно в [-1, 1].
    """
    if octaves < 1:
        raise ValueError("octaves должно быть >= 1")
    if persistence <= 0:
        raise ValueError("persistence должен быть больше нуля")
    if lacunarity <= 1.0:
        raise ValueError("lacunarity должен быть больше 1")

    total = np.zeros((height, width), dtype=np.float64)
    amplitude = 1.0
    amp_sum = 0.0
    freq_scale = float(scale)
    for i in range(octaves):
        # Для каждой октавы свой, но воспроизводимый seed
        octave = perlin_noise_2d(width, height, freq_scale, seed + i * 1000003)
        total += amplitude * octave
        amp_sum += amplitude
        amplitude *= persistence
        freq_scale *= lacunarity
    return total / amp_sum


def normalize(arr):
    """Приводит массив к диапазону [0, 1] (константа превращается в нули)."""
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def generate_heightmap(resolution=128, scale=4.0, seed=0, octaves=1,
                       persistence=0.5, lacunarity=2.0):
    """Карта высот (resolution x resolution), значения нормализованы в [0, 1].

    При octaves=1 это чистый шум Перлина (как на этапе 1), при octaves > 1 - fBm.
    """
    if not (MIN_RESOLUTION <= resolution <= MAX_RESOLUTION):
        raise ValueError(
            "Разрешение должно быть от %d до %d" % (MIN_RESOLUTION, MAX_RESOLUTION)
        )
    if octaves == 1:
        noise = perlin_noise_2d(resolution, resolution, scale=scale, seed=seed)
    else:
        noise = fbm_noise_2d(resolution, resolution, scale, octaves,
                             persistence, lacunarity, seed)
    return normalize(noise)
