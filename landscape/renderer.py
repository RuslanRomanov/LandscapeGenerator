"""Визуализация меша в Pygame: wireframe и закрашенная поверхность."""

import numpy as np
import pygame

BACKGROUND = (12, 16, 24)
LINE_COLOR = (90, 220, 130)


def _lod_indices(n, step):
    """Индексы узлов с шагом step; последний узел включается всегда."""
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return np.array(idx, dtype=np.int64)


def _project_grid(mesh, camera, w, h, r_idx, c_idx):
    """Проекция подсетки меша: экранные координаты (R, C, 2) int и видимость."""
    screen, visible = camera.project(mesh.vertices, w, h)
    screen = np.clip(screen, -10000, 10000)
    pts = np.rint(screen).astype(np.int32).reshape(mesh.rows, mesh.cols, 2)
    vis = visible.reshape(mesh.rows, mesh.cols)
    sub = np.ix_(r_idx, c_idx)
    return pts[sub], vis[sub]


def render_wireframe(surface, mesh, camera, line_step=2,
                     background=BACKGROUND, color=LINE_COLOR):
    """Рисует wireframe меша.

    line_step - рисовать каждую line_step-ю линию сетки; значение 1 рисует все.
    Если у меша есть цвета вершин (mesh.colors), каждый отрезок рисуется цветом
    своей начальной вершины, иначе используется color.
    """
    if line_step < 1:
        raise ValueError("line_step должен быть >= 1")

    w, h = surface.get_size()
    surface.fill(background)

    screen, visible = camera.project(mesh.vertices, w, h)
    screen = np.clip(screen, -10000, 10000)
    pts = np.rint(screen).astype(np.int32).reshape(mesh.rows, mesh.cols, 2)
    vis = visible.reshape(mesh.rows, mesh.cols)
    cols_rgb = mesh.color_grid()

    rows_to_draw = _lod_indices(mesh.rows, line_step)
    cols_to_draw = _lod_indices(mesh.cols, line_step)

    if cols_rgb is None:
        # Быстрый путь: ломаные одним цветом
        for r in rows_to_draw:
            _draw_polyline(surface, color, pts[r], vis[r])
        for c in cols_to_draw:
            _draw_polyline(surface, color, pts[:, c], vis[:, c])
        return

    # Цветной путь: отдельные отрезки. Рисуем от дальних к ближним, чтобы
    # ближние линии перекрывали дальние.
    segs_a = []
    segs_b = []
    segs_c = []
    segs_ok = []
    for r in rows_to_draw:
        segs_a.append(pts[r, :-1])
        segs_b.append(pts[r, 1:])
        segs_c.append(cols_rgb[r, :-1])
        segs_ok.append(vis[r, :-1] & vis[r, 1:])
    for c in cols_to_draw:
        segs_a.append(pts[:-1, c])
        segs_b.append(pts[1:, c])
        segs_c.append(cols_rgb[:-1, c])
        segs_ok.append(vis[:-1, c] & vis[1:, c])
    a = np.concatenate(segs_a)
    b = np.concatenate(segs_b)
    col = np.concatenate(segs_c)
    ok = np.concatenate(segs_ok)
    # Цвета вершин уже затемнены освещением; для тонких линий делаем ярче
    col = np.minimum(col.astype(np.int32) * 3 // 2 + 20, 255)
    a, b, col = a[ok].tolist(), b[ok].tolist(), col[ok].tolist()
    line = pygame.draw.line
    for pa, pb, pc in zip(a, b, col):
        line(surface, pc, pa, pb, 1)


def _draw_polyline(surface, color, pts, vis):
    """Рисует ломаную, пропуская участки с невидимыми точками."""
    start = 0
    n = len(pts)
    while start < n:
        if not vis[start]:
            start += 1
            continue
        end = start
        while end < n and vis[end]:
            end += 1
        if end - start >= 2:
            pygame.draw.lines(surface, color, False, pts[start:end].tolist(), 1)
        start = end


def render_solid(surface, mesh, camera, step=1, background=BACKGROUND):
    """Рисует закрашенную поверхность (алгоритм художника).

    Каждая ячейка сетки - четырёхугольник цвета (усреднение цветов вершин).
    Ячейки рисуются от дальних к ближним (по расстоянию до камеры в плоскости
    xz), чего достаточно для карты высот.
    step - шаг прореживания сетки (1 - полное разрешение, 2 - вдвое грубее и
    т. д.); используется для быстрой отрисовки во время вращения камеры.
    """
    if step < 1:
        raise ValueError("step должен быть >= 1")
    if mesh.colors is None:
        raise ValueError("Для закрашенной поверхности нужны цвета вершин")

    w, h = surface.get_size()
    surface.fill(background)

    r_idx = _lod_indices(mesh.rows, step)
    c_idx = _lod_indices(mesh.cols, step)
    pts, vis = _project_grid(mesh, camera, w, h, r_idx, c_idx)
    sub = np.ix_(r_idx, c_idx)
    rgb = mesh.color_grid()[sub].astype(np.float32)
    grid = mesh.grid()[sub]

    # Четыре угла каждой ячейки
    p00, p01 = pts[:-1, :-1], pts[:-1, 1:]
    p11, p10 = pts[1:, 1:], pts[1:, :-1]
    quad_pts = np.stack([p00, p01, p11, p10], axis=2)         # (R-1, C-1, 4, 2)
    quad_vis = vis[:-1, :-1] & vis[:-1, 1:] & vis[1:, 1:] & vis[1:, :-1]
    quad_rgb = (rgb[:-1, :-1] + rgb[:-1, 1:] + rgb[1:, 1:] + rgb[1:, :-1]) / 4.0

    # Порядок: от дальних к ближним
    centers = (grid[:-1, :-1] + grid[1:, 1:]) / 2.0
    cam = camera.position
    dist = (centers[..., 0] - cam[0]) ** 2 + (centers[..., 2] - cam[2]) ** 2
    order = np.argsort(-dist.ravel(), kind="stable")

    flat_pts = quad_pts.reshape(-1, 4, 2)[order]
    flat_rgb = np.rint(quad_rgb.reshape(-1, 3)[order]).astype(np.int32)
    flat_vis = quad_vis.ravel()[order]

    # Отбрасываем ячейки, целиком лежащие за пределами экрана
    xs, ys = flat_pts[..., 0], flat_pts[..., 1]
    on_screen = ((xs.max(axis=1) >= 0) & (xs.min(axis=1) <= w) &
                 (ys.max(axis=1) >= 0) & (ys.min(axis=1) <= h))
    keep = flat_vis & on_screen
    polys = flat_pts[keep].tolist()
    colors = flat_rgb[keep].tolist()

    polygon = pygame.draw.polygon
    for poly, col in zip(polys, colors):
        polygon(surface, col, poly)


def draw_hud(surface, lines):
    """Выводит строки текста в левом верхнем углу (только латиница)."""
    font = pygame.font.Font(None, 22)
    y = 8
    for text in lines:
        img = font.render(text, True, (230, 230, 230), None)
        shadow = font.render(text, True, (0, 0, 0))
        surface.blit(shadow, (9, y + 1))
        surface.blit(img, (8, y))
        y += img.get_height() + 2
