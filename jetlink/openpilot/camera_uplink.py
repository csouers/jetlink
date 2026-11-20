"""The comma's camera, for the host's dashboard.

The host gets no camera frames today: the warp runs on the comma's GPU and only its
393 KB output crosses the link (model_state.py: "the warp stays on the comma"). The
dashboard needs the frame itself, so this thread subscribes to camerad's vision IPC
alongside the model and the UI, keeps the newest frame, and downsamples it before the
link carries it.

Two things learned the hard way, both visible in this file:

* the frame goes at a fraction of the sensor's resolution (default 1/4). A full
  1344x760 NV12 frame is ~20 ms on the wire, which the model cannot pay inside
  HOLD_FRAME (46 ms): the large model handed straight back to the small one.
* nothing here may die quietly. `frames` and `error` are what a caller can read
  when the thread stops making progress, since its own log may not be visible.
"""
from __future__ import annotations

import threading
import time

import numpy as np

# narrow road 0, cabin 1, wide road 2 (openpilot's VisionStreamType order)
STREAMS = {"narrow_road": 0, "cabin": 1, "wide_road": 2}


class CameraUplink:
  """Newest NV12 frame of one camera, downsampled by `scale`, for the link."""

  def __init__(self, width: int, height: int, stride: int, stream: str = "wide_road",
               log=None, scale: int = 16):
    self.width, self.height, self.stride = width, height, stride
    self.scale = max(1, int(scale))
    self._stream = stream
    self._log = log
    self._frame: bytes | None = None
    self._at = 0.0
    self._lock = threading.Lock()
    self._stop = threading.Event()
    self._thread = threading.Thread(target=self._run, name="camera-uplink", daemon=True)
    self.frames = 0
    self.error: str | None = None

  def start(self) -> None:
    self._thread.start()

  def stop(self) -> None:
    self._stop.set()

  def latest(self, max_age: float = 0.5) -> tuple[bytes, float] | None:
    """(nv12 bytes, when it arrived), or None when there is nothing fresh."""
    with self._lock:
      if self._frame is None or time.monotonic() - self._at > max_age:
        return None
      return self._frame, self._at

  # -- the thread -------------------------------------------------------------

  def _run(self) -> None:
    try:
      from msgq.visionipc import VisionIpcClient
    except Exception as e:
      self.error = "no msgq.visionipc: %s" % e
      return

    stream_id = STREAMS.get(self._stream, STREAMS["wide_road"])
    client = VisionIpcClient("camerad", stream_id, True)
    warned = False
    while not self._stop.is_set():
      try:
        if not client.is_connected() and not client.connect(True):
          if not warned and self._log:
            self._log.warning("jetlink: camera uplink waiting for camerad %s", self._stream)
          warned = True
          time.sleep(0.5)
          continue
        warned = False
        buf = client.recv(200)
        if buf is not None:
          self._take(buf)
      except Exception as e:
        # keep the reason: a dead thread is otherwise invisible from outside
        self.error = "%s: %s" % (type(e).__name__, e)
        time.sleep(0.5)

  def _take(self, buf) -> None:
    """VisionBuf: Y plane then UV at uv_offset, both `stride` bytes a row."""
    w, h, stride = int(buf.width), int(buf.height), int(buf.stride)
    view = buf.data
    s = self.scale
    if s > 1:
      y = np.frombuffer(view, np.uint8, count=stride * h).reshape(h, stride)
      # the interleaved UV plane is stride * h/2 bytes: one U,V pair a pixel pair
      uv = np.frombuffer(view, np.uint8, count=stride * (h // 2),
                         offset=buf.uv_offset).reshape(h // 2, stride // 2, 2)
      y_s = np.ascontiguousarray(y[::s, ::s])
      uv_s = np.ascontiguousarray(uv[::s, ::s])
      frame = y_s.tobytes() + uv_s.tobytes()
      dims = (w // s, h // s, y_s.shape[1])
    else:
      y_len = stride * h
      uv_len = stride * (h // 2)
      frame = bytes(view[:y_len]) + bytes(view[buf.uv_offset:buf.uv_offset + uv_len])
      dims = (w, h, stride)
    with self._lock:
      self._frame = frame
      self._at = time.monotonic()
      self.width, self.height, self.stride = dims
    self.frames += 1
