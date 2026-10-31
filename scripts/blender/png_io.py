"""
png_io.py  —  8bit RGB PNG の自前読み書き（stdlib のみ）

protocol §13.4 / §14：Blender の画像保存を経由しない。
カラーマネジメントを通ると 8bit 値の保存が保証されず、
さらに Blender で書いて Blender で読むと変換が往復で打ち消され誤りを検出できない。

仕様を固定する
  - color type 2（truecolor RGB）/ bit depth 8
  - interlace なし
  - **filter type 0（None）を全 scanline に適用** … 往復で打ち消す変換を作らない
  - zlib level は既定（可逆なので画素値に影響しない）

numpy は任意。無い場合は bytearray 経路で動く（RunPod / Blender どちらでも可）。
"""
import struct
import zlib

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _chunk(tag, data):
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def write_rgb(path, rgb, width, height):
    """rgb: 長さ width*height*3 の 0-255 整数列（numpy 配列 / bytes / list）。
    行は上から下（PNG の走査順）。"""
    if np is not None and isinstance(rgb, np.ndarray):
        if rgb.dtype != np.uint8:
            raise ValueError("dtype must be uint8, got %s" % rgb.dtype)
        flat = rgb.reshape(-1)
        if flat.size != width * height * 3:
            raise ValueError("size mismatch: %d != %d"
                             % (flat.size, width * height * 3))
        raw = bytearray()
        row = width * 3
        for y in range(height):
            raw.append(0)                       # filter type 0
            raw += flat[y * row:(y + 1) * row].tobytes()
    else:
        buf = bytes(bytearray(rgb))
        if len(buf) != width * height * 3:
            raise ValueError("size mismatch: %d != %d"
                             % (len(buf), width * height * 3))
        raw = bytearray()
        row = width * 3
        for y in range(height):
            raw.append(0)
            raw += buf[y * row:(y + 1) * row]

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    out = (PNG_SIG + _chunk(b"IHDR", ihdr)
           + _chunk(b"IDAT", zlib.compress(bytes(raw)))
           + _chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(out)
    return len(out)


def read_rgb(path):
    """(pixels, width, height) を返す。pixels は bytes（RGB 連続、上から下）。
    filter 0 以外は明示的に拒否する（黙って別解釈しない）。"""
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != PNG_SIG:
        raise ValueError("not a PNG")
    pos = 8
    width = height = None
    idat = bytearray()
    while pos < len(data):
        (ln,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + ln]
        (crc,) = struct.unpack(">I", data[pos + 8 + ln:pos + 12 + ln])
        if crc != (zlib.crc32(tag + body) & 0xFFFFFFFF):
            raise ValueError("CRC mismatch in chunk %r" % tag)
        if tag == b"IHDR":
            width, height, depth, ctype, comp, filt, inter = \
                struct.unpack(">IIBBBBB", body)
            if (depth, ctype, comp, filt, inter) != (8, 2, 0, 0, 0):
                raise ValueError("unsupported PNG: depth=%d ctype=%d comp=%d "
                                 "filter=%d interlace=%d"
                                 % (depth, ctype, comp, filt, inter))
        elif tag == b"IDAT":
            idat += body
        elif tag == b"IEND":
            break
        pos += 12 + ln
    if width is None:
        raise ValueError("no IHDR")
    raw = zlib.decompress(bytes(idat))
    row = width * 3
    if len(raw) != (row + 1) * height:
        raise ValueError("IDAT size mismatch: %d != %d"
                         % (len(raw), (row + 1) * height))
    out = bytearray()
    for y in range(height):
        ft = raw[y * (row + 1)]
        if ft != 0:
            raise ValueError("filter type %d at row %d (0 only)" % (ft, y))
        out += raw[y * (row + 1) + 1:(y + 1) * (row + 1)]
    return bytes(out), width, height


def get_px(pixels, width, i, j_top):
    """PNG 走査順（j_top = 0 が上端）で 1 画素の (R,G,B) を返す。"""
    k = (j_top * width + i) * 3
    return pixels[k], pixels[k + 1], pixels[k + 2]
