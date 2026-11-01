"""Deterministic PNG fixtures for the image-input gate.

No third-party imports: zlib and struct only, so the same bytes are produced on
a laptop and in a Kaggle TPU container. Every fixture is 512x512 RGB, which is
the point: one image size means one vision-encoder compile, so the probe
measures the vision path instead of the recompile path. A traffic-frame study
must re-measure at the frame resolution it actually intends to send.

Answers are known by construction. Nothing here asks a model what is true;
that is the whole reason these fixtures exist.
"""

import hashlib
import struct
import zlib

SIZE = 512
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 30, 30)
GREEN = (30, 160, 60)
BLUE = (40, 60, 200)

# 5x7 bitmap font, one string per row, top to bottom.
FONT = {
    "A": "01110 10001 10001 11111 10001 10001 10001",
    "B": "11110 10001 10001 11110 10001 10001 11110",
    "C": "01110 10001 10000 10000 10000 10001 01110",
    "D": "11110 10001 10001 10001 10001 10001 11110",
    "E": "11111 10000 10000 11110 10000 10000 11111",
    "F": "11111 10000 10000 11110 10000 10000 10000",
    "G": "01110 10001 10000 10111 10001 10001 01111",
    "H": "10001 10001 10001 11111 10001 10001 10001",
    "I": "11111 00100 00100 00100 00100 00100 11111",
    "J": "00111 00010 00010 00010 00010 10010 01100",
    "K": "10001 10010 10100 11000 10100 10010 10001",
    "L": "10000 10000 10000 10000 10000 10000 11111",
    "M": "10001 11011 10101 10101 10001 10001 10001",
    "N": "10001 11001 10101 10011 10001 10001 10001",
    "O": "01110 10001 10001 10001 10001 10001 01110",
    "P": "11110 10001 10001 11110 10000 10000 10000",
    "Q": "01110 10001 10001 10001 10101 10010 01101",
    "R": "11110 10001 10001 11110 10100 10010 10001",
    "S": "01111 10000 10000 01110 00001 00001 11110",
    "T": "11111 00100 00100 00100 00100 00100 00100",
    "U": "10001 10001 10001 10001 10001 10001 01110",
    "V": "10001 10001 10001 10001 10001 01010 00100",
    "W": "10001 10001 10001 10101 10101 11011 10001",
    "X": "10001 10001 01010 00100 01010 10001 10001",
    "Y": "10001 10001 01010 00100 00100 00100 00100",
    "Z": "11111 00001 00010 00100 01000 10000 11111",
    "0": "01110 10001 10011 10101 11001 10001 01110",
    "1": "00100 01100 00100 00100 00100 00100 01110",
    "2": "01110 10001 00001 00010 00100 01000 11111",
    "3": "11111 00010 00100 00010 00001 10001 01110",
    "4": "00010 00110 01010 10010 11111 00010 00010",
    "5": "11111 10000 11110 00001 00001 10001 01110",
    "6": "00110 01000 10000 11110 10001 10001 01110",
    "7": "11111 00001 00010 00100 01000 01000 01000",
    "8": "01110 10001 10001 01110 10001 10001 01110",
    "9": "01110 10001 10001 01111 00001 00010 01100",
    " ": "00000 00000 00000 00000 00000 00000 00000",
    "-": "00000 00000 00000 11111 00000 00000 00000",
    ".": "00000 00000 00000 00000 00000 01100 01100",
}


class Canvas:
    """A mutable RGB raster with rectangles, circles and bitmap text."""

    def __init__(self, size=SIZE, background=WHITE):
        self.size = size
        self.px = bytearray(bytes(background) * (size * size))

    def _set(self, x, y, colour):
        if 0 <= x < self.size and 0 <= y < self.size:
            i = (y * self.size + x) * 3
            self.px[i : i + 3] = bytes(colour)

    def rect(self, x0, y0, x1, y1, colour):
        for y in range(max(0, y0), min(self.size, y1)):
            for x in range(max(0, x0), min(self.size, x1)):
                self._set(x, y, colour)

    def circle(self, cx, cy, r, colour):
        for y in range(max(0, cy - r), min(self.size, cy + r + 1)):
            dy = y - cy
            span = int((r * r - dy * dy) ** 0.5)
            for x in range(max(0, cx - span), min(self.size, cx + span + 1)):
                self._set(x, y, colour)

    def text(self, x, y, message, scale=8, colour=BLACK):
        for index, char in enumerate(message.upper()):
            rows = FONT.get(char, FONT[" "]).split()
            for dy, row in enumerate(rows):
                for dx, bit in enumerate(row):
                    if bit == "1":
                        self.rect(
                            x + (index * 6 + dx) * scale,
                            y + dy * scale,
                            x + (index * 6 + dx + 1) * scale,
                            y + (dy + 1) * scale,
                            colour,
                        )

    def text_size(self, message, scale=8):
        return len(message) * 6 * scale, 7 * scale

    def centred_text(self, message, scale=8, colour=BLACK):
        w, h = self.text_size(message, scale)
        self.text((self.size - w) // 2, (self.size - h) // 2, message, scale, colour)

    def png(self):
        """RGB8, filter type 0 on every scanline, one IDAT."""
        stride = self.size * 3
        raw = b"".join(
            b"\x00" + bytes(self.px[y * stride : (y + 1) * stride]) for y in range(self.size)
        )

        def chunk(tag, data):
            return (
                struct.pack(">I", len(data))
                + tag
                + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", self.size, self.size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b"")
        )


def _solid(colour):
    canvas = Canvas(background=colour)
    return canvas.png()


def _circles(points, radius=48, colour=BLUE, background=WHITE):
    canvas = Canvas(background=background)
    for cx, cy in points:
        canvas.circle(cx, cy, radius, colour)
    return canvas.png()


# Seven circles and three circles: the same question, two images, two answers.
# A text prior, a cached answer or a model that never saw the pixels cannot pass
# both, so this pair is the visual-dependence check.
CIRCLES_7 = [(128, 128), (256, 128), (384, 128), (128, 256), (256, 256), (128, 384), (256, 384)]
CIRCLES_3 = [(160, 160), (352, 160), (256, 352)]


# Two frames distinguishable by colour and by a numeral, for the image-order test.
# Deliberately bare: nothing in these two images but a background and a digit, so
# "the number printed in the first image" has exactly one referent and the answer
# cannot be confused with a count of anything else.
def _frame(background, numeral):
    canvas = Canvas(background=background)
    canvas.centred_text(numeral, scale=40, colour=WHITE)
    return canvas.png()


def _split(colour_left, colour_right):
    canvas = Canvas()
    canvas.rect(0, 0, SIZE // 2, SIZE, colour_left)
    canvas.rect(SIZE // 2, 0, SIZE, SIZE, colour_right)
    return canvas.png()


def _words(message, scale=12):
    canvas = Canvas()
    canvas.centred_text(message, scale=scale, colour=BLACK)
    return canvas.png()


def fixtures():
    """name -> PNG bytes. The order here is the order the probe reports."""
    return {
        "solid_red": _solid(RED),
        "solid_green": _solid(GREEN),
        "solid_blue": _solid(BLUE),
        "circles_7": _circles(CIRCLES_7),
        "circles_3": _circles(CIRCLES_3),
        "text_stop_7": _words("STOP 7"),
        "text_go_3": _words("GO 3"),
        "half_red_left": _split(RED, GREEN),
        "half_red_right": _split(GREEN, RED),
        "frame_1": _frame(RED, "1"),
        "frame_2": _frame(BLUE, "2"),
    }


def manifest(images=None):
    """Checksums and geometry for the record. Small enough to store beside results."""
    images = fixtures() if images is None else images
    return {
        "pixels": [SIZE, SIZE],
        "channels": 3,
        "note": "one resolution on purpose: a single vision-encoder compile for the whole battery",
        "fixtures": {
            name: {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
            for name, blob in images.items()
        },
    }


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    target = Path(sys.argv[1] if len(sys.argv) > 1 else "probe_images")
    target.mkdir(parents=True, exist_ok=True)
    built = fixtures()
    for name, blob in built.items():
        (target / f"{name}.png").write_bytes(blob)
    print(json.dumps(manifest(built), indent=2))
