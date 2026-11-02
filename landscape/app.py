"""Интерактивное окно Pygame: орбитальная камера, режимы отображения, экспорт."""

import os
import time

import numpy as np
import pygame

from .camera import Camera
from .export import export_obj
from .pipeline import build_landscape
from .renderer import render_solid, render_wireframe, draw_hud, BACKGROUND

MODE_SOLID = "solid"
MODE_WIRE = "wire"

ROTATE_SENSITIVITY = 0.4     # градусов на пиксель при перетаскивании мышью
KEY_ROTATE_STEP = 4.0        # градусов на нажатие стрелки
ZOOM_STEP = 0.9              # множитель расстояния на один щелчок колёсика
IDLE_BEFORE_FULL = 0.15      # секунд без движения до полной перерисовки
MESSAGE_TIME = 3.0           # сколько секунд показывать сообщение

# Клавиши-действия срабатывают один раз за нажатие. Автоповтор при удержании
# нужен только для вращения и масштаба (стрелки, + и -).
ACTION_KEYS = (pygame.K_ESCAPE, pygame.K_HOME, pygame.K_w, pygame.K_r,
               pygame.K_e, pygame.K_s)


class App:
    """Окно программы. Вся работа идёт через handle_event() и frame()."""

    def __init__(self, params, width=1000, height=700, mode=MODE_SOLID,
                 line_step=2, out_dir="."):
        self.params = params
        self.width = width
        self.height = height
        self.mode = mode
        self.line_step = line_step
        self.out_dir = out_dir

        self.screen = None
        self.scene = None            # кадр без текста
        self.landscape = None
        self.camera = None
        self.initial_view = None

        self.running = True
        self.dragging = False
        self.scene_dirty = True      # нужно перерисовать сцену
        self.full_pending = False    # после движения нужна полная перерисовка
        self.last_move = 0.0
        self.message = ""
        self.message_until = 0.0
        self.last_render_ms = 0.0
        self.last_full = True
        self.shot_index = 0
        self.clock = None
        self._msg_shown = None
        self._held_keys = set()      # зажатые клавиши-действия

    # ---------- подготовка ----------

    def setup(self):
        """Инициализирует Pygame, создаёт окно и первый ландшафт."""
        pygame.init()
        pygame.display.set_caption("Landscape generator - stage 2")
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.scene = pygame.Surface((self.width, self.height))
        pygame.key.set_repeat(250, 30)
        self.clock = pygame.time.Clock()
        self.generate(self.params)

    def generate(self, params):
        """Строит новый ландшафт и ставит камеру в исходное положение."""
        self.params = params
        self.landscape = build_landscape(params)
        self.camera = Camera.fit_to(self.landscape.mesh.vertices,
                                    yaw_deg=35.0, pitch_deg=35.0, margin=0.8)
        self.camera.min_distance = self.camera.distance * 0.35
        self.camera.max_distance = self.camera.distance * 3.0
        self.initial_view = (self.camera.yaw, self.camera.pitch,
                             self.camera.distance)
        self.scene_dirty = True
        self.full_pending = False

    def quit(self):
        pygame.quit()

    # ---------- события ----------

    def set_message(self, text):
        self.message = text
        self.message_until = time.time() + MESSAGE_TIME
        self._msg_shown = None   # принудительно перерисовать текст

    def reset_view(self):
        self.camera.yaw, self.camera.pitch, self.camera.distance = self.initial_view
        self._camera_changed()

    def _camera_changed(self):
        self.scene_dirty = True
        self.full_pending = True
        self.last_move = time.time()

    def new_seed(self, seed=None):
        if seed is None:
            seed = int(time.time() * 1000) % 1000000
        self.set_message("Generating seed %d ..." % seed)
        self.draw()   # показать сообщение до долгой операции
        self.generate(self.params.with_seed(seed))
        self.set_message("Seed %d, generated in %.0f ms"
                         % (seed, self.landscape.timings["total"] * 1000))

    def export(self):
        name = "landscape_seed%d.obj" % self.params.seed
        path = os.path.join(self.out_dir, name)
        nv, nf = export_obj(self.landscape.mesh, path)
        self.set_message("Saved %s + .mtl + .png (%d vertices)" % (name, nv))
        return path

    def screenshot(self):
        name = "landscape_seed%d_%d.png" % (self.params.seed, self.shot_index)
        path = os.path.join(self.out_dir, name)
        pygame.image.save(self.screen, path)
        self.shot_index += 1
        self.set_message("Saved %s" % name)
        return path

    def handle_event(self, event):
        """Обрабатывает одно событие Pygame."""
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in ACTION_KEYS:
                if event.key in self._held_keys:
                    return           # автоповтор при удержании - игнорируем
                self._held_keys.add(event.key)
            self._on_key(event.key)
        elif event.type == pygame.KEYUP:
            self._held_keys.discard(event.key)
        elif event.type in (pygame.WINDOWFOCUSLOST, pygame.WINDOWLEAVE):
            self._held_keys.clear()  # отпускание могло пройти мимо окна
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                dx, dy = event.rel
                self.camera.orbit(-dx * ROTATE_SENSITIVITY,
                                  dy * ROTATE_SENSITIVITY)
                self._camera_changed()
        elif event.type == pygame.MOUSEWHEEL:
            if event.y != 0:
                self.camera.zoom(ZOOM_STEP ** event.y)
                self._camera_changed()
        elif event.type == pygame.WINDOWEXPOSED:
            self.scene_dirty = True

    def _on_key(self, key):
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_LEFT:
            self.camera.orbit(-KEY_ROTATE_STEP, 0)
            self._camera_changed()
        elif key == pygame.K_RIGHT:
            self.camera.orbit(KEY_ROTATE_STEP, 0)
            self._camera_changed()
        elif key == pygame.K_UP:
            self.camera.orbit(0, KEY_ROTATE_STEP)
            self._camera_changed()
        elif key == pygame.K_DOWN:
            self.camera.orbit(0, -KEY_ROTATE_STEP)
            self._camera_changed()
        elif key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.camera.zoom(ZOOM_STEP)
            self._camera_changed()
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.camera.zoom(1.0 / ZOOM_STEP)
            self._camera_changed()
        elif key == pygame.K_HOME:
            self.reset_view()
        elif key == pygame.K_w:
            self.mode = MODE_WIRE if self.mode == MODE_SOLID else MODE_SOLID
            self.scene_dirty = True
        elif key == pygame.K_r:
            self.new_seed()
        elif key == pygame.K_e:
            self.export()
        elif key == pygame.K_s:
            self.screenshot()

    # ---------- отрисовка ----------

    def _fast_step(self):
        """Шаг прореживания сетки во время движения камеры."""
        return max(1, self.params.resolution // 64)

    def render_scene(self, full):
        """Рисует сцену в self.scene. full=False - упрощённо (быстро)."""
        mesh = self.landscape.mesh
        t = time.perf_counter()
        if self.mode == MODE_SOLID:
            render_solid(self.scene, mesh, self.camera,
                         1 if full else self._fast_step())
        else:
            step = self.line_step
            if not full:
                step = self.line_step * max(1, self.params.resolution // 128)
            render_wireframe(self.scene, mesh, self.camera, step)
        self.last_render_ms = (time.perf_counter() - t) * 1000.0
        self.last_full = full

    def hud_lines(self):
        p = self.params
        lines = [
            "Seed %d | %dx%d | octaves %d | erosion %d | water %.2f"
            % (p.seed, p.resolution, p.resolution, p.octaves,
               p.erosion_iterations, p.water_level),
            "Mode: %s | render %.0f ms%s"
            % ("solid" if self.mode == MODE_SOLID else "wireframe",
               self.last_render_ms, "" if self.last_full else " (fast)"),
            "Mouse drag - rotate, wheel - zoom, arrows - rotate, +/- - zoom, Home - reset",
            "W - wireframe/solid, R - new landscape, E - export OBJ, S - screenshot, Esc - quit",
        ]
        if self.message and time.time() < self.message_until:
            lines.append(self.message)
        return lines

    def draw(self):
        """Перерисовывает сцену (если нужно) и накладывает текст."""
        if self.scene_dirty:
            self.render_scene(full=not self.full_pending)
            self.scene_dirty = False
        self.screen.blit(self.scene, (0, 0))
        draw_hud(self.screen, self.hud_lines())
        pygame.display.flip()

    def frame(self):
        """Один кадр главного цикла: события, перерисовка. False - выход."""
        for event in pygame.event.get():
            self.handle_event(event)

        now = time.time()
        # После остановки камеры дорисовываем сцену в полном качестве
        if (self.full_pending and not self.scene_dirty
                and now - self.last_move > IDLE_BEFORE_FULL):
            self.full_pending = False
            self.scene_dirty = True

        message_visible = bool(self.message) and now < self.message_until
        if self.scene_dirty or message_visible != self._msg_shown:
            self._msg_shown = message_visible
            self.draw()
        return self.running

    def run(self):
        """Главный цикл программы."""
        self.setup()
        try:
            while self.frame():
                self.clock.tick(60)
        finally:
            self.quit()
        return 0
