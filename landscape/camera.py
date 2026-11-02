"""Орбитальная камера: вращение вокруг центра ландшафта и приближение."""

import math

import numpy as np


class Camera:
    """Перспективная камера, направленная на точку target.

    Положение камеры задаётся углами и расстоянием (сферические координаты).
    Методы orbit() и zoom() двигают камеру по сфере вокруг target, поэтому
    пролететь сквозь ландшафт нельзя: угол места ограничен, а расстояние
    лежит в пределах [min_distance, max_distance].
    Если не вызывать orbit() и zoom(), камера статична (как на этапе 1).
    """

    MIN_PITCH = 5.0    # градусы, чтобы не уходить под поверхность
    MAX_PITCH = 89.0

    def __init__(self, yaw_deg=35.0, pitch_deg=35.0, distance=100.0,
                 fov_deg=50.0, target=(0.0, 0.0, 0.0),
                 min_distance=None, max_distance=None):
        if distance <= 0:
            raise ValueError("distance должен быть больше нуля")
        if not (1.0 <= fov_deg <= 170.0):
            raise ValueError("fov_deg должен быть в диапазоне 1..170")
        self.yaw = math.radians(yaw_deg)
        self.pitch = math.radians(
            max(self.MIN_PITCH, min(self.MAX_PITCH, pitch_deg)))
        self.fov = math.radians(fov_deg)
        self.target = np.asarray(target, dtype=np.float32)
        self.near = 0.1
        self.min_distance = float(min_distance) if min_distance else distance * 0.2
        self.max_distance = float(max_distance) if max_distance else distance * 4.0
        if self.min_distance > self.max_distance:
            raise ValueError("min_distance больше max_distance")
        self.distance = float(
            min(max(distance, self.min_distance), self.max_distance))

    def orbit(self, d_yaw_deg, d_pitch_deg):
        """Поворот камеры вокруг цели на заданные углы (градусы)."""
        self.yaw = (self.yaw + math.radians(d_yaw_deg)) % (2.0 * math.pi)
        pitch = math.degrees(self.pitch) + d_pitch_deg
        self.pitch = math.radians(
            max(self.MIN_PITCH, min(self.MAX_PITCH, pitch)))

    def zoom(self, factor):
        """Умножает расстояние до цели на factor (меньше 1 - приближение)."""
        if factor <= 0:
            raise ValueError("factor должен быть больше нуля")
        self.distance = float(
            min(max(self.distance * factor, self.min_distance),
                self.max_distance))

    @classmethod
    def fit_to(cls, vertices, yaw_deg=35.0, pitch_deg=35.0, fov_deg=50.0,
               margin=1.05):
        """Создаёт камеру, при которой все вершины помещаются в кадр."""
        v = np.asarray(vertices, dtype=np.float32)
        lo = v.min(axis=0)
        hi = v.max(axis=0)
        target = (lo + hi) / 2.0
        radius = float(np.linalg.norm(hi - lo)) / 2.0
        distance = radius / math.sin(math.radians(fov_deg) / 2.0) * margin
        return cls(yaw_deg, pitch_deg, distance, fov_deg, target)

    @property
    def position(self):
        """Положение камеры в мировых координатах."""
        cp = math.cos(self.pitch)
        offset = np.array([
            math.sin(self.yaw) * cp,
            math.sin(self.pitch),
            math.cos(self.yaw) * cp,
        ], dtype=np.float32) * self.distance
        return self.target + offset

    def _basis(self):
        """Базис камеры: right, up, forward."""
        forward = self.target - self.position
        forward = forward / np.linalg.norm(forward)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        return right, up, forward

    def project(self, points, screen_w, screen_h):
        """Проецирует точки (N, 3) на экран.

        Возвращает (screen, visible):
            screen  - массив (N, 2) float32 с экранными координатами,
            visible - булев массив (N,): True, если точка перед камерой.
        """
        pts = np.asarray(points, dtype=np.float32)
        right, up, forward = self._basis()
        rel = pts - self.position

        xc = rel @ right
        yc = rel @ up
        zc = rel @ forward

        visible = zc > self.near
        safe_z = np.where(visible, zc, 1.0)

        focal = (screen_h / 2.0) / math.tan(self.fov / 2.0)
        sx = screen_w / 2.0 + focal * xc / safe_z
        sy = screen_h / 2.0 - focal * yc / safe_z
        screen = np.stack([sx, sy], axis=-1).astype(np.float32)
        return screen, visible
