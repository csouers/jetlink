"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

A ModelState whose policy runs on the Jetson.

Everything that touches the car stays on the comma: cameras, calibration, the
warp, the parser, controlsd, panda, CAN. The Jetson is a pure function, warped
frames and context in, 18452 floats out, and holds no control state.

The split is at `run_policy`. The warp stays on the comma's GPU: its input is a
2 MB camera buffer already there and its output the 393 KB the link carries
anyway. The history queues live on the Jetson, shipping them would cost ~10 MB
a frame instead of ~0.5 MB. modeld runs its warp as a JIT of its own, and
the link runs the same one; see warp.

A model that keeps its own history (openpilot #38916, Cinque Terre V3 on)
takes the same warped frame and the same scalars; its hidden state never
leaves the Jetson, so there is no prev_feat to send back. The spec says which.

What openpilot's modeld reads off a ModelState comes from the fork's adapter
(ModelFace): comma's constants, parser, smoothing and action function.
tinygrad is warp.Warp's to import: only modeld has it.
"""
from __future__ import annotations

import os
import struct
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from jetlink.protocol import NotFinite

# The dashboard's composite pane rides back with the frame that asked for it (see
# protocol Flag.WANT_PANE). It is for the screen, never for control, so every failure
# here is swallowed: a drive must not care whether a pane arrived.
PANE_ENDPOINT = 'paneView'
# msgq asserts 3 * message <= queue; a 386 KB pane needs a segment over 1.16 MB and
# an endpoint outside SERVICES gets a smaller default.
PANE_SEGMENT = 8 << 20
PANE_HDR = struct.Struct('<dQ')  # when it arrived, frame id, then the RGB pane
# Off unless asked for: publishing 386 KB costs modeld's thread up to ~4 ms a frame
# on a clock-capped comma.
PANE_FLAG_FILE = '/data/pane_view'

# The camera uplink: the comma's own frame rides in every other INFER_REQ (10 Hz at
# the model's 20) so the host's dashboard stops needing a replay. Off unless asked
# for, like the pane: 1.5 MB a frame at 10 Hz is real wire and real CPU.
CAMERA_FLAG_FILE = '/data/camera_uplink'
CAMERA_STRIDE = 1344
CAMERA_WIDTH = 1344
CAMERA_HEIGHT = 760
CAMERA_BYTES = CAMERA_WIDTH * CAMERA_HEIGHT * 3 // 2   # NV12
_camera_src = None


def _camera_hdr_for(src) -> bytes:
  from jetlink.protocol import pack_camera_hdr
  return pack_camera_hdr(src.width, src.height, src.stride)


def _camera_enabled() -> bool:
  return os.environ.get('JETLINK_CAMERA') == '1' or os.path.exists(CAMERA_FLAG_FILE)


_camera_err = None
CAMERA_STATS = '/data/camera_stats'
_stats_last = 0.0


def _camera_stats(**fields) -> None:
  """One JSON line a second: what the uplink is doing, for a link that cannot be
  read from the process that runs it."""
  global _stats_last
  now = time.monotonic()
  if now - _stats_last < 1.0:
    return
  _stats_last = now
  try:
    import json
    fields['t'] = round(now, 2)
    with open(CAMERA_STATS, 'w') as f:
      json.dump(fields, f)
  except Exception:
    pass


def _camera_source():
  """This process's camera thread, made once; None when it cannot be made, which

  retires the uplink for the process instead of failing frames."""
  global _camera_src
  if _camera_src is None:
    try:
      from jetlink.openpilot.camera_uplink import CameraUplink
      _camera_src = CameraUplink(CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_STRIDE)
      _camera_src.start()
    except Exception as e:
      global _camera_err
      import traceback
      _camera_err = "%s: %s" % (type(e).__name__, e)
      _camera_src = False
  return _camera_src or None

_pane_pub = None


def _pane_enabled() -> bool:
  return os.environ.get('JETLINK_PANE') == '1' or os.path.exists(PANE_FLAG_FILE)


def _pane_publisher():
  """This process's pane socket, made once. None when msgq cannot take it, which
  retires the pane for the process rather than failing a frame."""
  global _pane_pub
  if _pane_pub is None:
    try:
      import msgq
      _pane_pub = msgq.pub_sock(PANE_ENDPOINT, PANE_SEGMENT)
    except Exception:
      _pane_pub = False
  return _pane_pub or None

SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')
SLOW_FRAME = 0.05  # the full 20 Hz budget, not just the largest outliers
# How far into a frame end() waits for the reply before publishing the
# previous frame's output again (a held frame). Past this the camera frame
# would be dropped: modeld's own work around run() takes the rest of the 50 ms.
# A held frame is a plan one frame old, which the 10 s plan and the actuator
# delay both dwarf, where a dropped frame is the same stale plan plus a count
# toward selfdrived's modeldLagging. None waits the client's full deadline
HOLD_FRAME = 0.046
# Held frames the large model may make before it is behind rather than
# tailing: this many in a row, or more than HOLDS_ALLOWED within HOLD_WINDOW.
# A host at 40 ms a frame with a tail past 46 (an iPhone) holds a few frames
# in a hundred, each a plan one frame old; one at 60 ms holds every frame.
# More in a row than the client's deadline (link.INFERENCE_TIMEOUT, 0.2 s)
# lets a quiet host run to, so a run of holds is always a host that answers,
# late; one that has stopped answering fails the link first
# (JetlinkClient.infer_begin), as a lost link
HOLDS_IN_A_ROW = 5
HOLDS_ALLOWED = 20
HOLD_WINDOW = 10.0
# The first frames after a swap, which the fork holds engagement off for (a
# second at 20 Hz): a single held frame among them is behind. A host cold from
# a reconnect proves itself here, before anyone can engage, rather than at the
# wheel, where a slow host's return is a soft disable within a second
PROVING_FRAMES = 20
# the first of them never hand back, held or not: the first after a join
# carries the history reset, and a Mac's is ~100 ms of CoreML warm-up. A
# single slow frame later is the comma's own stall, a warp or a send, which
# handing back would not fix; only held frames say the host is behind
SETTLING_FRAMES = 3
# frames between asks for the server's telemetry, which rides on the response:
# every second one, as modeld sent a chestnut's state (20 Hz over 10 Hz)
TELEMETRY_EVERY = 2
# telemetry goes to the log at most this often
TELEMETRY_PERIOD = 1.0
# frames Trips keeps the timings of: a minute at 20 Hz
TRIPS_KEPT = 1200


class Trips:
  """What the comma measures of its frames over the link: the whole run, warp
  to parsed output, which is the frame as modeld waits on it, against the
  server's own total, which is all the server can see. Summarised for the
  leave (JetlinkClient.leave), so a phone's log carries the half of the frame
  budget it cannot measure. Bounded to the last TRIPS_KEPT frames."""

  def __init__(self):
    self.frames = 0          # frames waited for and published
    self.over = 0            # frames past SLOW_FRAME, the whole 20 Hz budget
    self.held = 0            # frames whose reply was late and published the previous one
    self.started = time.monotonic()
    self._whole_ms: deque[float] = deque(maxlen=TRIPS_KEPT)
    self._server_ms: deque[float] = deque(maxlen=TRIPS_KEPT)

  def record(self, whole_s: float, server_us: int) -> None:
    self.frames += 1
    if whole_s > SLOW_FRAME:
      self.over += 1
    self._whole_ms.append(whole_s * 1e3)
    self._server_ms.append(server_us / 1e3)

  def summary(self) -> dict:
    """frames, over, held, span_s, and over the frames kept: p50_ms, p99_ms,
    max_ms of the whole frame and server_ms, the server's mean total."""
    out = {'frames': self.frames, 'over': self.over, 'held': self.held,
           'span_s': round(time.monotonic() - self.started, 1)}
    if self._whole_ms:
      whole = sorted(self._whole_ms)
      out.update(p50_ms=round(whole[len(whole) // 2], 1), p99_ms=round(whole[int(0.99 * (len(whole) - 1))], 1),
                 max_ms=round(whole[-1], 1), server_ms=round(sum(self._server_ms) / len(self._server_ms), 1))
    return out


@dataclass
class Frame:
  """One frame on its way to the host: when its warp started, was launched
  and finished, then its seq and send time once sent. The bytes are the
  warp's output (Warp.output)."""
  t0: float
  t1: float
  t2: float
  seq: int | None = None
  sent: float = 0.0
  telemetry: bool = False


class JetlinkModelState:
  """Duck-types openpilot's modeld ModelState, and modeld_v2's where it differs.

  The small model is whatever bundle the user picked, on whichever modeld that
  bundle needs. sunnypilot's modeld_v2 reads constants, smoothing and the
  action function off the ModelState; stock modeld has them as module
  constants. This is comma's large model, so they are comma's.

  A frame is prepare(), send() and end(); run() is all three, as modeld calls
  it. The joining model (joining.JoiningModelState) reads `behind` after each
  frame.
  """

  prev_desire: np.ndarray  # for tracking the rising edge of the pulse

  def __init__(self, client, spec, warp, *, face, log, event):
    self._log = log
    # (name, **fields): a structured log line, cloudlog.event on a comma
    self._event = event
    self.face = face
    # the joining model sets it from the small model before the first frame,
    # and modeld writes it every frame after
    self.lat_delay = 0.0
    self.constants = face.constants
    self.LAT_SMOOTH_SECONDS = face.lat_smooth_seconds
    self.LONG_SMOOTH_SECONDS = face.long_smooth_seconds
    self.PLANPLUS_CONTROL = 1.0
    self.get_action_from_model = face.get_action_from_model

    self.client = client
    self.spec = spec
    # not chestnut hardware, but the same role: modelV2.big, the UI and the
    # model manager key off this flag
    self.chestnut = True
    # ask each frame for the dashboard's pane, and republish it to the UI process
    # (see PANE_*): read once, since a flip mid-drive would change the response
    # layout under a frame already in flight
    self._want_pane = _pane_enabled()
    # carry the comma's own camera to the host on every other frame (10 Hz at
    # the model's 20): the dashboard on the host replaces a replay feed with it
    self._want_camera = _camera_enabled()
    self._camera_tick = 0
    # The camera makes a request ~1.6 MB, and the join's proving frames cannot pay
    # that: measured, a 1.6 MB first frame took 82 ms and the model handed straight
    # back to the small one. So no camera until the link has run clean for a while,
    # and none for a while after any hold.
    self._camera_ok_from = 200

    # a warp.Warp or the adapter's model_warp, loaded and warmed ahead: its
    # first call links the JIT, and this runs on modeld's frame thread. Frames
    # go out of its output as they are, over either transport
    self.warp = warp

    self.input_shapes = spec.input_shapes
    self.output_slices = spec.output_slices
    # from the spec, not ModelConstants: the server derives its history stride
    # from the same field
    self.frame_skip = spec.frame_skip

    # compile_modeld.make_input_queues' packed_npy_inputs, minus the warp's
    # inputs and the GPU queues the server owns; for a stateful graph, minus
    # prev_feat too
    self.packed = np.zeros(spec.packed_nelem, dtype=np.float32)
    self.npy = {k: self.packed[at].reshape(shape) for k, (at, shape) in spec.packed_layout.items()}

    self.prev_desire = np.zeros(face.desire_len, dtype=np.float32)
    self.parser = face.parser()
    # the camera buffers modeld hands over, whatever the graph calls its inputs
    self.vision_input_names = ['img', 'big_img']
    self._need_reset = True
    self._frame_id = 0
    self._last_logged = 0.0
    self._last_hold_logged = 0.0
    self.trips = Trips()
    # the frame prepared and not yet ended, if any
    self._frame: Frame | None = None
    # the newest output parsed, with what it parsed to: a held frame
    # publishes the dict again rather than parsing the array a second time,
    # which it must not, the parser's softmax working in place
    self._parsed: tuple[np.ndarray, dict] | None = None
    # when the large model held a frame, within HOLD_WINDOW, and how many in
    # a row; `behind` says why it should hand back after the last frame
    self._holds: deque[float] = deque()
    self._holds_in_a_row = 0
    self.behind: str | None = None
    # whether the last frame published the one before it: the runtime record's
    self.held = False
    # a frame given up on, and when: its own reply was not back when its log
    # line was written, so _note_late_reply reports what it was answered with
    self._held_frame: int | None = None
    self._held_gave_up = 0.0

  def _send_camera(self) -> None:
    """Header + NV12 for this frame, on alternate frames, or None. Never raises:
    the uplink is a convenience for the host, not a driving input."""
    if self._frame_id % 200 == 0:
      self._log.warning("jetlink: camera dbg want=%s frame=%d ok_from=%d held=%s",
                        self._want_camera, self._frame_id, self._camera_ok_from, self.held)
    if not self._want_camera:
      return
    # The link is the model's first: no camera until it has run clean for a while,
    # and none for a while after any hold (a 1.6 MB frame inside the request once
    # took the first reply to 82 ms and the model handed straight back).
    if self._frame_id < self._camera_ok_from or self.held:
      return
    # Only on a link that is comfortably keeping up: a camera frame beside a slow
    # one is what took the sessions to p99 46 / max 1.5 s and the model behind.
    total_us = self.client.last_timings[2] if self.client.last_timings else 0
    if total_us > 30_000:
      return
    self._camera_tick += 1
    if self._camera_tick % 4:      # every fourth frame: 5 Hz at the model's 20
      return
    src = _camera_source()
    _camera_stats(stage='src', src=bool(src), frames=getattr(src, 'frames', None) if src else None,
                  err=getattr(src, 'error', None) if src else _camera_err)
    if src is None:
      self._log.warning("jetlink: camera uplink unavailable: %s", _camera_err)
      self._want_camera = False
      return
    got = src.latest()
    _camera_stats(stage='latest', got=bool(got), frames=getattr(src, 'frames', None))
    if got is None:
      if not getattr(self, '_camera_starved', False):
        self._camera_starved = True
        self._log.warning("jetlink: camera uplink has no frames (thread frames=%s connected=%s)",
                          getattr(src, 'frames', None), getattr(src, 'connected', None))
      return
    frame, _ = got
    ok = self.client.send_camera(_camera_hdr_for(src) + frame)
    _camera_stats(stage='send', ok=ok, bytes=len(frame), frame_id=self._frame_id)
    if ok and not getattr(self, '_camera_logged', False):
      self._camera_logged = True
      self._log.warning("jetlink: camera uplink on, %d bytes a frame (%dx%d stride %d)",
                        len(frame) + 12, src.width, src.height, src.stride)

  def _publish_pane(self) -> None:
    """The pane this frame's reply carried, to the UI. Never raises: it is a view."""
    pane = self.client.last_pane
    if pane is None:
      return
    pub = _pane_publisher()
    if pub is None:
      self._want_pane = False
      return
    frame_id, received_at, rgb = pane
    if not rgb:
      return   # an empty slice: the server sent the count and no bytes this frame
    try:
      pub.send(PANE_HDR.pack(received_at, frame_id) + bytes(rgb))
    except Exception as e:
      self._want_pane = False
      self._log.warning('jetlink: pane publish failed, dropping panes: %s', e)

  def slice_outputs(self, model_outputs: np.ndarray, output_slices: dict[str, slice]) -> dict[str, np.ndarray]:
    return {k: model_outputs[np.newaxis, v] for k, v in output_slices.items()}

  def log_telemetry(self) -> None:
    """The server's health, piggybacked on the previous response, to the log
    at 1 Hz. chestnutState is comma's board on the wire and a Jetson's
    telemetry has no message of its own yet."""
    telemetry = self.client.last_state
    if not telemetry:
      return
    now = time.monotonic()
    if now - self._last_logged < TELEMETRY_PERIOD:
      return
    self._last_logged = now
    self._event("jetlinkTelemetry", dead=bool(self.client.dead), **telemetry)
    send = getattr(self.client.t, 'last_send', None)
    if send:
      self._event("jetlinkSend", nonce=self.client.nonce, frame=self._frame_id, totals=self.client.t.send_totals.copy(), **send)

  def prepare(self, bufs: dict, transforms: dict[str, np.ndarray], inputs: dict[str, np.ndarray]) -> None:
    """Take this frame up: start the warp, and while the GPU runs it, read
    what the host has answered so far (a held frame's late reply) and pack
    the rest of the inputs. send() puts it on the link and end() waits for
    its answer."""
    t0 = time.perf_counter()
    self.warp.start(_address(bufs['img']), _address(bufs['big_img']), transforms['img'], transforms['big_img'])
    t1 = time.perf_counter()
    # the GPU is warping; nothing below is an input to it
    self.client.drain()
    self.log_telemetry()

    # Model decides when action is completed, so desire input is just a pulse triggered on rising edge.
    # Under whichever name the loop keyed it: stock modeld's desire_pulse, or a modeld_v2 bundle's own
    desire = inputs[next(k for k in inputs if k.startswith('desire'))]
    desire[0] = 0
    self.npy['desire'][:] = np.where(desire - self.prev_desire > .99, desire, 0)
    self.prev_desire[:] = desire
    self.npy['traffic_convention'][:] = inputs['traffic_convention']
    self.npy['action_t'][:] = inputs['action_t']

    # free again once infer_begin returns: the gadget's io_submit and the
    # cable's sendmsg have copied it by then, before the next warp writes it
    self.warp.wait()
    self._frame = Frame(t0, t1, time.perf_counter())

  def send(self, want_telemetry: bool = False) -> None:
    """The prepared frame to the host. It asks for the server's telemetry
    when the log is due for it, or when told to: modeld asks on the frames it
    would have sent a chestnut's state on.

    With an output to hold, a frame the link cannot take without waiting for
    the host to drain earlier ones is not sent, and end() holds it: the same
    plan one frame old a late reply publishes, where waiting stalls the frame
    loop on a host that is behind."""
    frame = self._frame
    self._frame_id += 1
    frame.telemetry = want_telemetry or time.monotonic() - self._last_logged >= TELEMETRY_PERIOD
    try:
      frame.seq = self.client.infer_begin(self.warp.output, self.packed, self._frame_id, reset=self._need_reset,
                                          want_state=frame.telemetry, skip_if_busy=self._can_hold,
                                          want_pane=self._want_pane)
    except Exception:
      self._log.warning("jetlink: frame %d send failed: %s", self._frame_id, getattr(self.client.t, 'last_send', {}))
      raise
    frame.sent = time.perf_counter()
    if frame.seq is not None:
      self._need_reset = False

  def end(self, after_enqueue: Callable[[], None] | None = None) -> dict[str, np.ndarray]:
    """The sent frame's output, as modeld blocks on a chestnut's, but not past
    HOLD_FRAME once there is an output to publish again: a frame that long
    is a dropped camera frame. Only a stall past the client's deadline, or an
    output that is not finite, raises, into the joining state's demotion to
    the small model."""
    frame, self._frame = self._frame, None
    # publish health while the Jetson works
    if after_enqueue is not None:
      after_enqueue()
    waiting_from = time.perf_counter()
    if frame.seq is None:
      model_output = None   # not sent (see send): held
    else:
      try:
        model_output = self.client.infer_end(frame.seq, hold=self._hold_left(frame) if self._can_hold else None)
      except NotFinite:
        # answered, so the leave's summary counts it
        self.trips.record(time.perf_counter() - frame.t0, self.client.last_timings[2])
        raise
    if self._want_pane and model_output is not None:
      self._publish_pane()
    self._send_camera()
    t4 = time.perf_counter()
    held = self.held = model_output is None
    if held and self._want_camera:
      self._camera_ok_from = self._frame_id + 100
    self.behind = self._note_hold(held)
    if held:
      self._held_frame, self._held_gave_up = self._frame_id, t4
    self.trips.record(t4 - frame.t0, self.client.last_timings[2])
    outputs = self._outputs(self.client.last_output if held else model_output)
    # a frame past the budget is a dropped camera frame; send against reply
    # says which end it was. Held frames are a steady few on a marginal host,
    # so after the first they go to the log at TELEMETRY_PERIOD
    log_hold = held and (self.trips.held <= 3 or t4 - self._last_hold_logged >= TELEMETRY_PERIOD)
    if self._frame_id <= 3 or t4 - frame.t0 > SLOW_FRAME or log_hold:
      note = ''
      if held:
        self._last_hold_logged = t4
        note = f', held ({self.trips.held} so far{", not sent" if frame.seq is None else ""})'
      # persisted on the comma so a drive can separate server execution from
      # receive stalls once the Jetson is offline; server total excludes USB
      gpu_us, queue_us, total_us = self.client.last_timings
      # a held frame has no answer yet: last_timings there is whatever reply
      # was read last, not this frame's, and _note_late_reply says what this
      # one was answered with once it lands
      server = ('; its reply is not back yet' if held
                else f'; server gpu {gpu_us / 1e3:.1f} queue {queue_us / 1e3:.1f} total {total_us / 1e3:.1f} ms')
      receive = getattr(self.client.t, 'last_receive', {})
      # over the cable: how frames go, how many never came back, and what the
      # kernel dropped on the way in (softnet), since the link came up
      drops = getattr(self.client.t, 'net_drops', lambda: None)()
      cable = (f"; frames {'as datagrams' if self.client.t.datagrams else 'on the stream'}, "
               f"{self.client.frames_lost} lost, net drops {drops}" if drops is not None else '')
      self._log.warning("jetlink: frame %d warp %.1f data %.1f send %.1f wait %.1f ms%s%s; "
                        "ffs maxima prepare %.1f read_wait %.1f handoff %.1f ms%s",
                        self._frame_id, (frame.t1 - frame.t0) * 1e3, (frame.t2 - frame.t1) * 1e3,
                        (frame.sent - frame.t2) * 1e3, (t4 - waiting_from) * 1e3,
                        note, server,
                        receive.get('prepare', 0.0) * 1e3,
                        receive.get('read_wait', 0.0) * 1e3, receive.get('handoff', 0.0) * 1e3, cable)
    self._note_late_reply()
    return outputs

  def _note_late_reply(self) -> None:
    """The held frame's own answer, once it lands.

    A hold gives up at HOLD_FRAME, so the frame's reply is read by whatever
    reads next. How long after the give-up it came decides what the fix is:
    a reply that lands a few ms late carried a server total inside the frame
    budget, which is bytes or a wakeup between the two clocks and not a host
    that cannot keep up; one that lands tens of ms late is the host.
    """
    if self._held_frame is None:
      return
    reply = getattr(self.client, 'reply_for', lambda _fid: None)(self._held_frame)
    if reply is None:
      return
    held, late_ms = self._held_frame, (reply.landed_at - self._held_gave_up) * 1e3
    self._held_frame = None
    self._log.warning("jetlink: held frame %d was answered %.1f ms after the hold gave up: "
                      "server gpu %.1f queue %.1f total %.1f ms, %.1f ms after its send",
                      held, late_ms, reply.gpu_us / 1e3, reply.queue_us / 1e3, reply.total_us / 1e3,
                      (reply.landed_at - reply.sent_at) * 1e3)

  def run(self, bufs: dict, transforms: dict[str, np.ndarray],
          inputs: dict[str, np.ndarray], after_enqueue: Callable[[], None] | None = None) -> dict[str, np.ndarray]:
    self.prepare(bufs, transforms, inputs)
    self.send(after_enqueue is not None)
    return self.end(after_enqueue)

  @property
  def _can_hold(self) -> bool:
    """Is there an output to publish again instead of waiting (HOLD_FRAME)?"""
    return bool(HOLD_FRAME) and self.client.last_output is not None

  @staticmethod
  def _hold_left(frame) -> float:
    """What is left of the hold for `frame`, from its warp's start."""
    return HOLD_FRAME - (time.perf_counter() - frame.t0)

  def _note_hold(self, held: bool) -> str | None:
    """Why the large model should hand back after this frame, if it should:
    any held frame among its first PROVING_FRAMES past SETTLING_FRAMES,
    HOLDS_IN_A_ROW held frames running, or more than HOLDS_ALLOWED in
    HOLD_WINDOW. None while it is keeping up."""
    if not held:
      self._holds_in_a_row = 0
      return None
    self.trips.held += 1
    now = time.monotonic()
    self._holds.append(now)
    while now - self._holds[0] > HOLD_WINDOW:
      self._holds.popleft()
    self._holds_in_a_row += 1
    if self._frame_id <= SETTLING_FRAMES:
      return None
    if self._frame_id <= PROVING_FRAMES:
      return f'held frame {self._frame_id} of the first {PROVING_FRAMES}'
    if self._holds_in_a_row >= HOLDS_IN_A_ROW:
      return f'held {self._holds_in_a_row} frames in a row'
    if len(self._holds) > HOLDS_ALLOWED:
      return f'held {len(self._holds)} frames in {HOLD_WINDOW:.0f} s'
    return None

  def _outputs(self, model_output: np.ndarray) -> dict[str, np.ndarray]:
    """The output as modeld reads it. An array parsed before (a held frame's)
    gives the same dict again: the parser's softmax works in place on it."""
    if self._parsed is not None and self._parsed[0] is model_output:
      return self._parsed[1]
    # the server checks first (Status.NOT_FINITE, which the client raises as
    # this). This is for one that did not, or bytes the link changed: a raw
    # inf is a corrupt output even where the parser would clip it finite. A
    # sum allocates nothing, and any NaN or inf makes it one; these outputs
    # are far from overflow
    if not np.isfinite(model_output.sum()):
      raise NotFinite(f"frame {self._frame_id}: the output is not finite")
    outputs_dict = self.parser.parse_outputs(self.slice_outputs(model_output, self.output_slices))
    # a finite output can still parse to inf, by a parser that does not clip
    if not np.isfinite(sum(float(value.sum()) for value in outputs_dict.values())):
      raise NotFinite(f"frame {self._frame_id}: the parsed output is not finite")
    if SEND_RAW_PRED:
      outputs_dict['raw_pred'] = model_output.copy()
    self._parsed = (model_output, outputs_dict)
    return outputs_dict

  def close(self) -> None:
    """Let go of the link; the joining state calls this on a model it retires."""
    self.client.close()


def _address(buf) -> int:
  """Where a camera buffer modeld hands over (a VisionBuf) keeps its bytes."""
  return np.frombuffer(buf.data, dtype=np.uint8).ctypes.data
