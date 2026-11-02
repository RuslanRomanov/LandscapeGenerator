"""Гидравлическая эрозия (модель "труб", Mei et al.), векторизованная на NumPy.

Каждая ячейка карты хранит слой воды и взвешенные наносы. На каждой итерации:
дождь -> потоки воды между соседними ячейками -> новая глубина воды и скорость ->
размыв или осаждение наносов -> перенос наносов потоком -> испарение.
Все шаги - операции над целыми массивами, циклов по пикселям нет.
"""

import numpy as np


def _shift(a, dy, dx):
    """Сдвиг массива с заполнением нулями: результат[y, x] = a[y - dy, x - dx]."""
    out = np.zeros_like(a)
    h, w = a.shape
    ys_dst = slice(max(dy, 0), h + min(dy, 0))
    ys_src = slice(max(-dy, 0), h + min(-dy, 0))
    xs_dst = slice(max(dx, 0), w + min(dx, 0))
    xs_src = slice(max(-dx, 0), w + min(-dx, 0))
    out[ys_dst, xs_dst] = a[ys_src, xs_src]
    return out


def _bilinear(a, ys, xs):
    """Билинейная выборка массива a в дробных координатах (с ограничением по краям)."""
    h, w = a.shape
    ys = np.clip(ys, 0.0, h - 1.0)
    xs = np.clip(xs, 0.0, w - 1.0)
    y0 = np.minimum(np.floor(ys).astype(np.int64), h - 2)
    x0 = np.minimum(np.floor(xs).astype(np.int64), w - 2)
    fy = ys - y0
    fx = xs - x0
    top = a[y0, x0] * (1 - fx) + a[y0, x0 + 1] * fx
    bot = a[y0 + 1, x0] * (1 - fx) + a[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bot * fy


def hydraulic_erosion(heightmap, iterations=60, relief=0.3, seed=0,
                      rain=0.012, evaporation=0.03, capacity=1.0,
                      dissolve=0.1, deposit=0.1, gravity=9.8, dt=0.1,
                      max_depth=1.0, min_tilt=0.05, max_speed=4.0):
    """Применяет гидравлическую эрозию к карте высот.

    heightmap  - массив (R, C) со значениями в [0, 1];
    iterations - число шагов симуляции (0 - вернуть копию без изменений);
    relief     - отношение максимальной высоты к ширине карты: задаёт крутизну
                 рельефа в единицах размера ячейки (0.3 - как в визуализации);
    seed       - зерно для случайной неравномерности дождя;
    rain, evaporation, capacity, dissolve, deposit - параметры модели.

    Возвращает новую карту высот того же размера, нормализованную в [0, 1].
    Исходный массив не изменяется.
    """
    hm = np.asarray(heightmap, dtype=np.float64)
    if hm.ndim != 2 or hm.shape[0] < 4 or hm.shape[1] < 4:
        raise ValueError("Карта высот должна быть двумерной, минимум 4 x 4")
    if iterations < 0:
        raise ValueError("iterations должно быть >= 0")
    if iterations == 0:
        return hm.copy()

    rows, cols = hm.shape
    # Рабочая высота в единицах размера ячейки
    h = hm * (relief * max(rows, cols))
    lo_in = float(hm.min())
    hi_in = float(hm.max())

    rng = np.random.default_rng(seed)
    rain_map = rain * rng.uniform(0.5, 1.5, size=hm.shape)

    d = np.zeros_like(h)      # глубина воды
    s = np.zeros_like(h)      # наносы во взвеси
    fl = np.zeros_like(h)     # потоки наружу: влево, вправо, вверх, вниз
    fr = np.zeros_like(h)
    fu = np.zeros_like(h)
    fd = np.zeros_like(h)

    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    eps = 1e-9

    for _ in range(iterations):
        # 1. Дождь
        d += rain_map * dt

        # 2. Потоки воды в соседние ячейки (по разности полного уровня воды)
        total = h + d
        tp = np.pad(total, 1, mode="edge")
        fl = np.maximum(0.0, fl + dt * gravity * (total - tp[1:-1, :-2]))
        fr = np.maximum(0.0, fr + dt * gravity * (total - tp[1:-1, 2:]))
        fu = np.maximum(0.0, fu + dt * gravity * (total - tp[:-2, 1:-1]))
        fd = np.maximum(0.0, fd + dt * gravity * (total - tp[2:, 1:-1]))
        # Через границу карты вода не вытекает
        fl[:, 0] = 0.0
        fr[:, -1] = 0.0
        fu[0, :] = 0.0
        fd[-1, :] = 0.0
        # Нельзя вытечь больше воды, чем есть в ячейке
        out_sum = fl + fr + fu + fd
        k = np.minimum(1.0, d / (out_sum * dt + eps))
        fl *= k
        fr *= k
        fu *= k
        fd *= k
        out_sum = fl + fr + fu + fd

        # 3. Новая глубина воды и скорость течения
        in_l = _shift(fr, 0, 1)    # то, что течёт к нам слева (их вправо)
        in_r = _shift(fl, 0, -1)
        in_u = _shift(fd, 1, 0)
        in_d = _shift(fu, -1, 0)
        d_new = d + dt * (in_l + in_r + in_u + in_d - out_sum)
        d_new = np.maximum(d_new, 0.0)
        d_avg = 0.5 * (d + d_new)
        wx = 0.5 * (in_l - fl + fr - in_r)
        wy = 0.5 * (in_u - fu + fd - in_d)
        wet = d_avg > 1e-4
        denom = np.where(wet, d_avg, 1.0)
        vx = np.where(wet, wx / denom, 0.0)
        vy = np.where(wet, wy / denom, 0.0)
        speed = np.sqrt(vx * vx + vy * vy)
        scale_v = np.minimum(1.0, max_speed / (speed + eps))
        vx *= scale_v
        vy *= scale_v
        speed *= scale_v
        d = d_new

        # 4. Размыв и осаждение
        gy, gx = np.gradient(h)
        sin_a = np.sqrt(gx * gx + gy * gy) / np.sqrt(1.0 + gx * gx + gy * gy)
        sin_a = np.maximum(sin_a, min_tilt)
        depth_factor = np.clip(d / max_depth, 0.0, 1.0)
        cap = capacity * sin_a * speed * depth_factor
        diff = cap - s
        erode = diff > 0
        amount = np.where(erode, dissolve * diff, deposit * diff)
        # Нельзя унести больше, чем есть, и нельзя осадить больше, чем во взвеси
        amount = np.minimum(amount, np.maximum(h, 0.0))
        h = h - amount
        s = s + amount

        # 5. Перенос наносов течением
        s = _bilinear(s, yy - vy * dt, xx - vx * dt)

        # 6. Испарение
        d *= (1.0 - evaporation * dt)

    # Нормализация обратно в [0, 1] с сохранением исходного размаха
    span = hi_in - lo_in
    if span < 1e-12:
        return hm.copy()
    out = h / (relief * max(rows, cols))
    omin, omax = float(out.min()), float(out.max())
    if omax - omin < 1e-12:
        return np.zeros_like(hm)
    return (out - omin) / (omax - omin)
