import Foundation

/// A dense 3-class mask of the road ahead, rasterized from the model's own
/// parametric lane output.
///
/// openpilot's supercombo does not emit a segnet any more: `SEGNET_SIZE` in
/// `openpilot/common/transformations/model.py` is a leftover of the pre-2023
/// model, and the reply has no room for it. What it emits, in `lane_lines`
/// and `road_edges`, is the same geometry as polynomials in the **calib
/// frame**: 4 lane lines and 2 road edges, each 33 points of (y, z) sampled at
/// the fixed `X_IDXS` distances, plus the standard deviation of each. Drawing
/// those into the image the model was fed recovers a mask shaped like the old
/// segnet, for which the only inputs are floats the reply already carries.
///
/// The projection needs no calibration — and so nothing new on the wire. The
/// model's own raster is a pinhole camera at zero roll, pitch and yaw relative
/// to the calib frame (`get_view_frame_from_calib_frame(0, 0, 0, 0)`), which
/// makes `K_model @ view_from_device_frame` a constant per model size:
///
///   px = w/2 + fl * y / x        py = cy + fl * z / x
///
/// The same constants openpilot renders training labels with, so a lane line
/// lands where the model expects it rather than where a fitted camera would
/// put it.
public enum LaneMask {
  /// One byte a pixel: none, or one of these.
  public enum Cell: UInt8 {
    case none = 0
    /// Between the outermost lane lines: the drivable surface.
    case road = 1
    /// A lane line, drawn on top of the road.
    case lane = 2
    /// A road edge, drawn on top of the road.
    case edge = 3
  }

  /// openpilot's `ModelConstants.IDX_N` and `X_IDXS`: 33 distances, quadratic
  /// from 0 to 192 m, at which `lane_lines` and `road_edges` are sampled.
  public static let idxN = 33
  public static let xIdxs: [Double] = (0..<idxN).map { 192.0 * pow(Double($0) / Double(idxN - 1), 2) }

  /// `NUM_LANE_LINES`, `NUM_ROAD_EDGES`, `LANE_LINES_WIDTH`: (y, z) a point.
  public static let laneLines = 4
  public static let roadEdges = 2
  public static let lineWidth = 2

  /// The floats one line's `mu` takes: 33 points of (y, z).
  public static let lineFloats = idxN * lineWidth

  /// The pinhole a model raster is: intrinsics folded with the fixed
  /// view-to-device rotation, so only (y, z, x) of a calib-frame point matter.
  public struct Projection: Sendable, Equatable {
    public let width: Int
    public let height: Int
    /// Focal length in pixels, and the principal point.
    public let focal: Double
    public let cx: Double
    public let cy: Double

    public init(width: Int, height: Int, focal: Double, cx: Double, cy: Double) {
      self.width = width
      self.height = height
      self.focal = focal
      self.cx = cx
      self.cy = cy
    }

    /// The model sizes openpilot's `transformations/model.py` names: MEDMODEL
    /// and BIGMODEL. Both share a focal length; the big raster only doubles,
    /// and its principal point sits at `256 + MEDMODEL_CY`, not twice the small
    /// one, because it is openpilot's own framing rather than an upscale.
    /// `SBIGMODEL` (fl 455) is the wide camera's frame and never holds lanes.
    public static let medmodel = Projection(width: 512, height: 256, focal: 910.0, cx: 256.0, cy: 47.6)
    public static let bigmodel = Projection(width: 1024, height: 512, focal: 910.0, cx: 512.0, cy: 256.0 + 47.6)

    /// A calib-frame point to a pixel in this raster, or nil when it is
    /// behind the camera (the polynomials reach back to x = 0, and the first
    /// few sample distances are behind or at the lens) or not a number.
    public func pixel(x: Double, y: Double, z: Double) -> (x: Double, y: Double)? {
      guard x > 0.05, x.isFinite, y.isFinite, z.isFinite else { return nil }
      let px = cx + focal * y / x
      let py = cy + focal * z / x
      guard px.isFinite, py.isFinite else { return nil }
      return (px, py)
    }
  }

  /// The raster a spec's frames are in. Both ends of the warp agree without
  /// being told: the warped planes are (2, 6, h, w), six quarter-size planes
  /// of one packed YUV420 image, so the image is (2h, 2w) and its shape
  /// picks the projection.
  public static func projection(for spec: ModelSpec) -> Projection? {
    let (h, w) = spec.modelHW
    guard h > 0, w > 0 else { return nil }
    let image = (width: 2 * w, height: 2 * h)
    switch image {
    case (Projection.medmodel.width, Projection.medmodel.height): return .medmodel
    case (Projection.bigmodel.width, Projection.bigmodel.height): return .bigmodel
    default: return nil
    }
  }

  /// What a reply has to carry for a mask: the polylines, in the reply's own
  /// float layout, copied out so the render can run after the frame's buffer
  /// is free again.
  public struct Slices: Sendable, Equatable {
    /// 4 lines of 33 (y, z), then 2 edges of 33 (y, z).
    public var points: [Float]
    public init(points: [Float]) { self.points = points }
  }

  /// Pull `lane_lines` and `road_edges` out of a reply: `outputs` is the whole
  /// driving output, `count` floats of it (hidden_state included — it is all
  /// the engine wrote). Nil when the layout names neither slice.
  public static let laneSliceName = "lane_lines"
  public static let edgeSliceName = "road_edges"

  public static func slices(_ outputs: UnsafePointer<Float>, count: Int, spec: ModelSpec) -> Slices? {
    func floats(_ name: String, _ wanted: Int) -> [Float]? {
      guard let range = spec.outputSlices.first(where: { $0.name == name })?.range(in: count),
        range.count >= wanted, range.upperBound <= count
      else { return nil }
      return Array(UnsafeBufferPointer(start: outputs + range.lowerBound, count: wanted))
    }
    guard let lanes = floats(laneSliceName, laneLines * lineFloats),
      let edges = floats(edgeSliceName, roadEdges * lineFloats)
    else { return nil }
    return Slices(points: lanes + edges)
  }

  /// A mask of this projection's size, all `none`.
  public static func empty(_ projection: Projection) -> [UInt8] {
    [UInt8](repeating: Cell.none.rawValue, count: projection.width * projection.height)
  }

  /// Rasterize a reply's polylines into a mask.
  ///
  /// Road is filled between the outermost lines that project, since at four
  /// lines the outermost pair is the whole visible surface; lanes and edges
  /// are stroked on top, and a fill never overwrites a stroke.
  public static func render(_ slices: Slices, projection: Projection, lineWidth: Int = 2) -> [UInt8] {
    var mask = empty(projection)
    let stride = projection.width

    func line(_ index: Int, of base: Int) -> [(x: Double, y: Double)] {
      let at = (base + index) * lineFloats
      guard at + lineFloats <= slices.points.count else { return [] }
      var out: [(x: Double, y: Double)] = []
      out.reserveCapacity(idxN)
      for i in 0..<idxN {
        let y = Double(slices.points[at + i * lineWidth])
        let z = Double(slices.points[at + i * lineWidth + 1])
        if let p = projection.pixel(x: xIdxs[i], y: y, z: z) { out.append(p) }
      }
      return out
    }

    let lanes = (0..<laneLines).map { line($0, of: 0) }
    let edges = (0..<roadEdges).map { line($0, of: laneLines) }

    // A fill needs two lines with something to connect: one point is a ray
    // from the camera, and the polynomial's near samples are behind it.
    func usable(_ points: [(x: Double, y: Double)]) -> Bool { points.count >= 2 }

    let present = (0..<laneLines).filter { usable(lanes[$0]) }
    if let first = present.first, let last = present.last, first != last {
      var polygon = lanes[first]
      polygon += lanes[last].reversed()
      fill(&mask, stride: stride, height: projection.height, polygon: polygon, value: Cell.road.rawValue)
    }

    for points in edges where usable(points) {
      stroke(&mask, stride: stride, height: projection.height, points: points, width: max(1, lineWidth), value: Cell.edge.rawValue)
    }
    for points in lanes where usable(points) {
      stroke(&mask, stride: stride, height: projection.height, points: points, width: max(1, lineWidth), value: Cell.lane.rawValue)
    }
    return mask
  }

  // MARK: drawing

  /// Even-odd scanline fill, clipped to the mask. Points far outside are kept:
  /// the clip is per span, so a polygon that leaves the frame still fills the
  /// part of it that is on screen.
  static func fill(_ mask: inout [UInt8], stride: Int, height: Int, polygon: [(x: Double, y: Double)], value: UInt8) {
    guard polygon.count >= 3 else { return }
    var minY = Int.max, maxY = Int.min
    for p in polygon {
      minY = min(minY, Int(p.y.rounded(.down)))
      maxY = max(maxY, Int(p.y.rounded(.up)))
    }
    minY = max(minY, 0)
    maxY = min(maxY, height - 1)
    guard minY <= maxY else { return }

    var xs: [Double] = []
    xs.reserveCapacity(polygon.count)
    for row in minY...maxY {
      let y = Double(row) + 0.5
      xs.removeAll(keepingCapacity: true)
      var j = polygon.count - 1
      for i in 0..<polygon.count {
        let a = polygon[i], b = polygon[j]
        if (a.y <= y) != (b.y <= y) {
          let t = (y - a.y) / (b.y - a.y)
          xs.append(a.x + t * (b.x - a.x))
        }
        j = i
      }
      guard xs.count >= 2 else { continue }
      xs.sort()
      var k = 0
      while k + 1 < xs.count {
        let from = max(0, Int(xs[k].rounded(.down)))
        let to = min(stride - 1, Int(xs[k + 1].rounded(.up)))
        if from <= to {
          let base = row * stride
          for x in from...to { mask[base + x] = value }
        }
        k += 2
      }
    }
  }

  /// A thick polyline: every pixel whose centre is within half a width of a
  /// segment, over the segment's bounding box only.
  static func stroke(_ mask: inout [UInt8], stride: Int, height: Int, points: [(x: Double, y: Double)], width: Int, value: UInt8) {
    guard points.count >= 2 else { return }
    let radius = Double(max(width, 1)) * 0.5
    for i in 0..<(points.count - 1) {
      let a = points[i], b = points[i + 1]
      let loX = max(0, Int((min(a.x, b.x) - radius).rounded(.down)))
      let hiX = min(stride - 1, Int((max(a.x, b.x) + radius).rounded(.up)))
      let loY = max(0, Int((min(a.y, b.y) - radius).rounded(.down)))
      let hiY = min(height - 1, Int((max(a.y, b.y) + radius).rounded(.up)))
      guard loX <= hiX, loY <= hiY else { continue }
      let dx = b.x - a.x, dy = b.y - a.y
      let lengthSquared = dx * dx + dy * dy
      for row in loY...hiY {
        let base = row * stride
        let py = Double(row) + 0.5
        for col in loX...hiX {
          let px = Double(col) + 0.5
          var distance: Double
          if lengthSquared <= 0 {
            distance = ((px - a.x) * (px - a.x) + (py - a.y) * (py - a.y)).squareRoot()
          } else {
            var t = ((px - a.x) * dx + (py - a.y) * dy) / lengthSquared
            t = min(max(t, 0), 1)
            let qx = a.x + t * dx, qy = a.y + t * dy
            distance = ((px - qx) * (px - qx) + (py - qy) * (py - qy)).squareRoot()
          }
          if distance <= radius && mask[base + col] != Cell.lane.rawValue {
            mask[base + col] = value
          }
        }
      }
    }
  }
}

/// Where a rendered mask goes: the file this machine's consumers read, as the
/// pane and the camera frame have one. Off unless asked for — a mask is for an
/// experiment, and 130 KB twenty times a second is disk nobody asked to spend.
///
///   echo half > ~/.jetlink-mask-mode        # full | half | off
///   echo /path/mask.bin > ~/.jetlink-mask-file
///
/// `half` is every other frame (10 Hz), `full` every reply.
public enum LaneMaskFile {
  public static let defaultPath = "/tmp/jetlink-mask.bin"

  public static var modeURL: URL {
    URL(fileURLWithPath: NSHomeDirectory() + "/.jetlink-mask-mode")
  }

  public static var pathURL: URL {
    URL(fileURLWithPath: NSHomeDirectory() + "/.jetlink-mask-file")
  }

  public enum Mode: String {
    case off, half, full
  }

  public static func mode() -> Mode {
    let text = (try? String(contentsOf: modeURL, encoding: .utf8))?
      .trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? "off"
    return Mode(rawValue: text) ?? .off
  }

  public static func path() -> String {
    guard let cfg = try? String(contentsOf: pathURL, encoding: .utf8) else { return defaultPath }
    let named = cfg.trimmingCharacters(in: .whitespacesAndNewlines)
    return named.isEmpty ? defaultPath : named
  }

  /// Whether this frame's mask is due, by the mode file. Read once a frame:
  /// the file is a few bytes and the server already does this for the pane.
  public static func due(_ frameID: UInt32) -> Bool {
    switch mode() {
    case .off: return false
    case .half: return frameID & 1 == 0
    case .full: return true
    }
  }

  /// A 16-byte header — arrival time in ms, frame id, and (height << 16 | width)
  /// so a reader needs nothing but the file — then one byte a pixel. Replaced
  /// atomically, so a reader never sees half a mask.
  public static func write(_ mask: [UInt8], projection: LaneMask.Projection, frameID: UInt32) {
    var data = Data(capacity: mask.count + 16)
    var stamp = DispatchTime.now().uptimeNanoseconds / 1_000_000
    withUnsafeBytes(of: &stamp) { data.append(contentsOf: $0) }
    var fid = frameID
    withUnsafeBytes(of: &fid) { data.append(contentsOf: $0) }
    var dims = UInt32((projection.height << 16) | (projection.width & 0xFFFF))
    withUnsafeBytes(of: &dims) { data.append(contentsOf: $0) }
    data.append(contentsOf: mask)
    try? data.write(to: URL(fileURLWithPath: path()), options: .atomic)
  }
}
