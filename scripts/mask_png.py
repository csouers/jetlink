#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory of this source tree.

Look at the lane mask the Mac server draws from a reply's lane lines.

The server writes it when `~/.jetlink-mask-mode` says so (`full` or `half`),
to `/tmp/jetlink-mask.bin` unless `~/.jetlink-mask-file` names another path:
a 16-byte header, then one byte a pixel — 0 none, 1 road, 2 lane, 3 road edge.

    python3 scripts/mask_png.py                      # /tmp/jetlink-mask.bin -> mask.png
    python3 scripts/mask_png.py --out view.png       # somewhere else
    python3 scripts/mask_png.py --repeat             # one PNG a second, to watch it drive

No dependencies: the PNG is written with zlib and struct, so this runs on the
comma as readily as on the Mac.
"""
import argparse
import os
import struct
import sys
import time
import zlib

HEADER = struct.Struct('<QII')  # arrival ms, frame id, (height << 16) | width
COLORS = {0: (0, 0, 0), 1: (0, 80, 0), 2: (0, 255, 0), 3: (255, 128, 0)}


def read(path):
  with open(path, 'rb') as f:
    blob = f.read()
  stamp, frame_id, dims = HEADER.unpack_from(blob, 0)
  width, height = dims & 0xFFFF, dims >> 16
  body = blob[HEADER.size:]
  if len(body) != width * height:
    raise SystemExit(f"{path}: {len(body)} bytes, but {width}x{height} says {width * height}")
  return stamp, frame_id, width, height, body


def png(width, height, body):
  rows = bytearray()
  for y in range(height):
    rows.append(0)  # filter: none
    base = y * width
    for cell in body[base:base + width]:
      rows.extend(COLORS.get(cell, (255, 0, 255)))

  def chunk(kind, data):
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xFFFFFFFF)

  return (b'\x89PNG\r\n\x1a\n'
          + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
          + chunk(b'IDAT', zlib.compress(bytes(rows), 6))
          + chunk(b'IEND', b''))


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--mask', default=os.path.expanduser('~/.jetlink-mask-file'))
  parser.add_argument('--out', default='mask.png')
  parser.add_argument('--repeat', action='store_true', help='rewrite the PNG whenever the mask file moves')
  parser.add_argument('--quiet', action='store_true')
  args = parser.parse_args()

  path = os.path.expanduser(args.mask)
  if not os.path.exists(path):
    path = '/tmp/jetlink-mask.bin'

  last = None
  while True:
    try:
      stamp, frame_id, width, height, body = read(path)
    except FileNotFoundError:
      print(f"no mask at {path}: turn it on with `echo half > ~/.jetlink-mask-mode`", file=sys.stderr)
      return 1
    except SystemExit as e:
      print(e, file=sys.stderr)
      return 1

    if args.repeat and stamp == last:
      time.sleep(0.05)
      continue
    last = stamp

    with open(args.out, 'wb') as f:
      f.write(png(width, height, body))
    if not args.quiet:
      counts = {k: body.count(k) for k in COLORS}
      total = width * height
      print(f"frame {frame_id} {width}x{height} age {(time.time() * 1000 - stamp) / 1000:.1f}s -> {args.out} "
            + ' '.join(f"{name}={counts[cell] * 100 / total:.1f}%" for cell, name in
                       zip((1, 2, 3), ('road', 'lane', 'edge'))))
    if not args.repeat:
      return 0


if __name__ == '__main__':
  sys.exit(main())
