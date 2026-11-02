"""Процедурный генератор 3D-ландшафта - этап 2.

Шум (fBm) -> гидравлическая эрозия -> раскраска по высоте -> 3D-меш.
Просмотр в окне Pygame с орбитальной камерой, экспорт в OBJ.

Примеры запуска:
    python main.py
    python main.py --seed 42 --resolution 192 --erosion 150
    python main.py --seed 7 --export-obj land.obj --save-image land.png --no-window
"""

import argparse
import os
import sys

from landscape.noise import MIN_RESOLUTION, MAX_RESOLUTION
from landscape.pipeline import TerrainParams


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Процедурный генератор 3D-ландшафта (этап 2)")
    p.add_argument("--seed", type=int, default=0, help="зерно генератора (0)")
    p.add_argument("--resolution", type=int, default=128,
                   help="разрешение карты высот, %d..%d (128)"
                   % (MIN_RESOLUTION, MAX_RESOLUTION))
    p.add_argument("--scale", type=float, default=3.0,
                   help="число ячеек шума вдоль стороны карты (3)")
    p.add_argument("--octaves", type=int, default=5, help="число октав fBm (5)")
    p.add_argument("--persistence", type=float, default=0.5,
                   help="затухание амплитуды по октавам (0.5)")
    p.add_argument("--lacunarity", type=float, default=2.0,
                   help="рост частоты по октавам (2.0)")
    p.add_argument("--erosion", type=int, default=100, metavar="N",
                   help="число итераций гидравлической эрозии, 0 - выключить (100)")
    p.add_argument("--water-level", type=float, default=0.3,
                   help="уровень воды, 0..1 (0.3)")
    p.add_argument("--height-scale", type=float, default=30.0,
                   help="множитель высоты рельефа (30)")
    p.add_argument("--mode", choices=("solid", "wire"), default="solid",
                   help="режим отображения: solid или wire (solid)")
    p.add_argument("--line-step", type=int, default=2,
                   help="в режиме wire рисовать каждую N-ю линию сетки (2)")
    p.add_argument("--width", type=int, default=1000, help="ширина окна")
    p.add_argument("--height", type=int, default=700, help="высота окна")
    p.add_argument("--out-dir", default=".",
                   help="папка для файлов, сохраняемых из окна (OBJ, PNG)")
    p.add_argument("--export-obj", metavar="ФАЙЛ", default=None,
                   help="сохранить меш в OBJ и продолжить работу")
    p.add_argument("--save-image", metavar="ФАЙЛ", default=None,
                   help="сохранить кадр в PNG")
    p.add_argument("--no-window", action="store_true",
                   help="не открывать окно (вместе с --save-image/--export-obj)")
    args = p.parse_args(argv)

    if not (MIN_RESOLUTION <= args.resolution <= MAX_RESOLUTION):
        p.error("--resolution должен быть от %d до %d"
                % (MIN_RESOLUTION, MAX_RESOLUTION))
    if args.scale <= 0:
        p.error("--scale должен быть больше нуля")
    if args.octaves < 1:
        p.error("--octaves должен быть >= 1")
    if args.persistence <= 0:
        p.error("--persistence должен быть больше нуля")
    if args.lacunarity <= 1.0:
        p.error("--lacunarity должен быть больше 1")
    if args.erosion < 0:
        p.error("--erosion должен быть >= 0")
    if not (0.0 <= args.water_level < 1.0):
        p.error("--water-level должен быть в диапазоне [0, 1)")
    if args.height_scale <= 0:
        p.error("--height-scale должен быть больше нуля")
    if args.line_step < 1:
        p.error("--line-step должен быть >= 1")
    if args.width < 200 or args.height < 200:
        p.error("размер окна слишком мал")
    return args


def make_params(args):
    return TerrainParams(
        resolution=args.resolution, seed=args.seed, scale=args.scale,
        octaves=args.octaves, persistence=args.persistence,
        lacunarity=args.lacunarity, erosion_iterations=args.erosion,
        water_level=args.water_level, height_scale=args.height_scale)


def main(argv=None):
    args = parse_args(argv)
    params = make_params(args)

    if args.no_window:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # pygame импортируем после выбора видеодрайвера
    from landscape.app import App

    app = App(params, args.width, args.height, mode=args.mode,
              line_step=args.line_step, out_dir=args.out_dir)

    if args.no_window or args.save_image or args.export_obj:
        app.setup()
        try:
            app.draw()
            if args.export_obj:
                from landscape.export import export_obj
                nv, nf = export_obj(app.landscape.mesh, args.export_obj)
                print("OBJ сохранён: %s (%d вершин, %d треугольников)"
                      % (args.export_obj, nv, nf))
            if args.save_image:
                import pygame
                pygame.image.save(app.screen, args.save_image)
                print("Кадр сохранён: %s" % args.save_image)
            print("Генерация: %s" % ", ".join(
                "%s %.0f мс" % (k, v * 1000)
                for k, v in app.landscape.timings.items()))
            if args.no_window:
                return 0
            # Окно остаётся открытым: продолжаем в интерактивном режиме
            while app.frame():
                app.clock.tick(60)
        finally:
            app.quit()
        return 0

    return app.run()


if __name__ == "__main__":
    sys.exit(main())
