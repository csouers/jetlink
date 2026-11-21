import Foundation
import JetlinkKit
import JetlinkTestSupport
import Testing

@testable import JetlinkServer

/// The mask openpilot's parametric lanes rasterize to: the constants are
/// openpilot's own (`transformations/model.py`, `modeld/constants.py`), so a
/// wrong number puts the road somewhere the model never looks.
@Suite("Lane mask")
struct LaneMaskTests {
  /// The projection is a pinhole with the fixed view-to-device rotation folded
  /// in: px = w/2 + fl*y/x, py = cy + fl*z/x. These are the numbers numpy gives
  /// for the same matrix, so a change to either end is caught.
  @Test("Med and big projections match openpilot's intrinsics")
  func projections() {
    let med = LaneMask.Projection.medmodel
    #expect(med.width == 512 && med.height == 256)
    #expect(med.focal == 910.0 && med.cx == 256.0 && med.cy == 47.6)

    // straight ahead, at the camera's height: the vanishing point
    let vp = try! #require(med.pixel(x: 20, y: 0, z: 0))
    #expect(abs(vp.x - 256) < 1e-9 && abs(vp.y - 47.6) < 1e-9)

    // 20 m ahead, 1.8 m to the right, 1.22 m below the camera
    let p = try! #require(med.pixel(x: 20, y: 1.8, z: 1.22))
    #expect(abs(p.x - 337.9) < 0.05)
    #expect(abs(p.y - 103.11) < 0.05)

    // the big model doubles the raster and keeps the focal length, so its
    // principal point is 256 + MEDMODEL_CY rather than 2 * MEDMODEL_CY
    let big = try! #require(LaneMask.Projection.bigmodel.pixel(x: 20, y: 1.8, z: 1.22))
    #expect(abs(big.x - 593.9) < 0.05)
    #expect(abs(big.y - 359.11) < 0.05)

    // behind the camera is not a pixel
    #expect(med.pixel(x: 0, y: 0, z: 0) == nil)
    #expect(med.pixel(x: -5, y: 1, z: 1) == nil)
  }

  /// `X_IDXS` is 192*(i/32)^2: the samples bunch up near the car, where a
  /// metre of road is a lot of pixels.
  @Test("X_IDXS is the quadratic openpilot samples lanes at")
  func xIdxs() {
    #expect(LaneMask.xIdxs.count == 33)
    #expect(LaneMask.xIdxs[0] == 0)
    #expect(abs(LaneMask.xIdxs[1] - 0.1875) < 1e-12)
    #expect(abs(LaneMask.xIdxs[8] - 12.0) < 1e-12)
    #expect(abs(LaneMask.xIdxs[16] - 48.0) < 1e-12)
    #expect(abs(LaneMask.xIdxs[32] - 192.0) < 1e-12)
  }

  /// A spec's raster size comes from its warped planes alone: six quarter-size
  /// planes of one packed YUV420 image, so the image is (2h, 2w).
  @Test("The raster follows the spec, and unknown sizes get no mask")
  func rasterFromSpec() {
    #expect(LaneMask.projection(for: spec(modelHW: (128, 256))) == .medmodel)
    #expect(LaneMask.projection(for: spec(modelHW: (256, 512))) == .bigmodel)
    #expect(LaneMask.projection(for: spec(modelHW: (64, 128))) == nil)
  }

  /// Straight lane lines 1.8 m either side of the car, 1.22 m below the camera
  /// (openpilot's calib frame: x forward, y right, z down), 20 m out. Row 103
  /// is x = 20 m ahead, and the road must span the three-and-a-half metres
  /// either side of centre that the outermost pair implies.
  @Test("Road fills between the outermost lines, lanes stroke on top")
  func render() {
    let projection = LaneMask.Projection.medmodel
    // y = -3.6, -1.8, +1.8, +3.6: outer pair fills, inner pair strokes
    let ys = [-3.6, -1.8, 1.8, 3.6]
    var points: [Float] = []
    for y in ys {
      for _ in 0..<LaneMask.idxN {
        points.append(Float(y))
        points.append(1.22)
      }
    }
    // road edges a little further out
    for y in [-4.2, 4.2] {
      for _ in 0..<LaneMask.idxN {
        points.append(Float(y))
        points.append(1.22)
      }
    }

    let mask = LaneMask.render(LaneMask.Slices(points: points), projection: projection)
    #expect(mask.count == projection.width * projection.height)

    let row = 103
    func at(_ x: Int) -> LaneMask.Cell { LaneMask.Cell(rawValue: mask[row * projection.width + x])! }

    #expect(at(256) == .road)
    #expect(at(200) == .road)
    // the right-hand inner lane line lands at 256 + 910*1.8/20 = 337.9
    #expect(at(338) == .lane)
    #expect(at(174) == .lane)
    // the right road edge at 256 + 910*4.2/20 = 447.1
    #expect(at(447) == .edge)
    // beyond every line is nothing
    #expect(at(20) == .none)
    #expect(at(500) == .none)
    // above the vanishing point the polynomials project out of the frame or
    // behind it, so the top rows stay empty
    #expect(LaneMask.Cell(rawValue: mask[0])! == .none)
  }

  /// The reply carries the polylines in its own layout, so they are read by
  /// slice name — and a layout without them yields no mask rather than a
  /// corrupted one.
  @Test("Slices are read by name out of the driving output")
  func slicesByName() {
    let layout = spec(modelHW: (128, 256), slices: [
      ("lane_lines", 0, 528), ("road_edges", 536, 800),
    ])
    var outputs = [Float](repeating: 0, count: 1000)
    for i in 0..<528 { outputs[i] = Float(i) }
    for i in 536..<800 { outputs[i] = Float(i) }

    // each slice holds `mu` then `std`: parse_mdn halves it, and only the
    // first half is a point. 4 lanes and 2 edges of 33 (y, z) is 396 floats.
    let slices = try! #require(LaneMask.slices(outputs, count: outputs.count, spec: layout))
    #expect(slices.points.count == 264 + 132)
    #expect(slices.points[0] == 0)
    #expect(slices.points[264] == 536)

    let without = spec(modelHW: (128, 256), slices: [("meta", 0, 55)])
    #expect(LaneMask.slices(outputs, count: outputs.count, spec: without) == nil)
  }

  // MARK: fixtures

  private func spec(modelHW: (Int, Int), slices: [(String, Int, Int)] = []) -> ModelSpec {
    let (h, w) = modelHW
    return ModelSpec(
      sha256: String(repeating: "0", count: 64), nbytes: 1, frameSkip: 4,
      inputShapes: [NamedShape("new_img", [2, 6, h, w])],
      outputShapes: [NamedShape("outputs", [1000])],
      outputSlices: slices.map { NamedSlice($0.0, start: $0.1, stop: $0.2) },
      checkpoint: nil)
  }
}
