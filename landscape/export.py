"""Экспорт меша в формат Wavefront OBJ (с материалом и текстурой цвета)."""

import os
import struct
import zlib

import numpy as np


def write_png(path, rgb):
    """Записывает массив (H, W, 3) uint8 в PNG без внешних библиотек."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    # К каждой строке добавляется байт фильтра 0 (без фильтрации)
    raw = np.concatenate([np.zeros((h, 1), dtype=np.uint8),
                          rgb.reshape(h, w * 3)], axis=1).tobytes()

    def chunk(tag, data):
        body = tag + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


def export_obj(mesh, path, object_name="landscape", with_colors=True,
               with_texture=True):
    """Сохраняет меш в OBJ-файл и, при with_texture, файлы .mtl и .png.

    Записываются вершины (v), нормали (vn), текстурные координаты (vt) и
    треугольники (f). Для регулярной сетки индексы v, vt и vn совпадают, грани
    имеют вид "f a/a/a b/b/b c/c/c". Индексы в OBJ начинаются с 1.
    Цвета берутся без освещения (mesh.albedo; если его нет - mesh.colors).
    Если with_colors=True, цвет дополнительно пишется в строки вершин
    ("v x y z r g b", значения 0..1).
    Возвращает кортеж (число вершин, число треугольников).
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64) + 1
    colors = mesh.albedo if mesh.albedo is not None else mesh.colors
    use_colors = with_colors and colors is not None
    use_texture = with_texture and colors is not None
    use_normals = mesh.normals is not None

    base = os.path.splitext(path)[0]
    stem = os.path.basename(base)
    mtl_name = stem + ".mtl"
    tex_name = stem + "_color.png"
    folder = os.path.dirname(path)

    if use_texture:
        grid = np.asarray(colors, dtype=np.uint8).reshape(mesh.rows, mesh.cols, 3)
        write_png(os.path.join(folder, tex_name), grid)
        with open(os.path.join(folder, mtl_name), "w", encoding="ascii",
                  newline="\n") as f:
            f.write("# Material of the procedural landscape\n")
            f.write("newmtl landscape_material\n")
            f.write("Ka 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\n")
            f.write("Ns 10.0\nd 1.0\nillum 1\n")
            f.write("map_Kd %s\n" % tex_name)

    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("# Procedural landscape generator\n")
        f.write("# vertices: %d, triangles: %d\n" % (len(verts), len(faces)))
        if use_texture:
            f.write("mtllib %s\n" % mtl_name)
        f.write("o %s\n" % object_name)

        if use_colors:
            rgb = np.asarray(colors, dtype=np.float64) / 255.0
            np.savetxt(f, np.hstack([verts, rgb]),
                       fmt="v %.5f %.5f %.5f %.4f %.4f %.4f")
        else:
            np.savetxt(f, verts, fmt="v %.5f %.5f %.5f")

        if use_texture:
            # Центры текселей совпадают с вершинами; v=0 - низ картинки
            u = (np.arange(mesh.cols) + 0.5) / mesh.cols
            v = 1.0 - (np.arange(mesh.rows) + 0.5) / mesh.rows
            uu, vv = np.meshgrid(u, v)
            np.savetxt(f, np.column_stack([uu.ravel(), vv.ravel()]),
                       fmt="vt %.6f %.6f")
        if use_normals:
            np.savetxt(f, np.asarray(mesh.normals, dtype=np.float64),
                       fmt="vn %.5f %.5f %.5f")
        if use_texture:
            f.write("usemtl landscape_material\n")
            f.write("s 1\n" if use_normals else "s off\n")

        a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
        if use_texture and use_normals:
            np.savetxt(f, np.column_stack([a, a, a, b, b, b, c, c, c]),
                       fmt="f %d/%d/%d %d/%d/%d %d/%d/%d")
        elif use_texture:
            np.savetxt(f, np.column_stack([a, a, b, b, c, c]),
                       fmt="f %d/%d %d/%d %d/%d")
        elif use_normals:
            np.savetxt(f, np.column_stack([a, a, b, b, c, c]),
                       fmt="f %d//%d %d//%d %d//%d")
        else:
            np.savetxt(f, faces, fmt="f %d %d %d")
    return len(verts), len(faces)
