import Foundation

/// The comma's camera frame, as the request carries it (see Wire.Flag.camera).
///
/// Written to a file for this machine's own consumers — the dashboard reads it as a
/// live source instead of needing a replay — with a 12-byte header (arrival
/// milliseconds, frame id) in front of the NV12 the comma sent. `os.replace`-style
/// atomic writes, so a reader never sees half a frame.
///
/// Kept out of Session.swift so the whole thing is one small unit: what it does on
/// the wire is the session's business, what lands on disk is this file's.
enum CameraIngest {
  static let headerBytes = 12
  /// Counted for the log; the session writes it, nothing reads it for control.
  nonisolated(unsafe) static var frames = 0

  private static let lock = NSLock()

  static var path: String {
    if let cfg = try? String(contentsOf: URL(fileURLWithPath: NSHomeDirectory() + "/.jetlink-camera-file"),
                            encoding: .utf8) {
      let named = cfg.trimmingCharacters(in: .whitespacesAndNewlines)
      if !named.isEmpty { return named }
    }
    return "/tmp/jetlink-camera.bin"
  }

  /// Store one frame. Called from the session's receive thread, so the file is only
  /// ever replaced whole and only one writer runs at a time.
  static func store(_ blob: UnsafeRawBufferPointer, frameID: UInt32) {
    var data = Data(capacity: blob.count + headerBytes)
    // wall clock, not since-boot: the dashboard compares it with time.time()
    var stamp = UInt64(max(0, Date().timeIntervalSince1970 * 1000))
    withUnsafeBytes(of: &stamp) { data.append(contentsOf: $0) }
    var fid = frameID
    withUnsafeBytes(of: &fid) { data.append(contentsOf: $0) }
    data.append(contentsOf: blob)
    lock.lock()
    defer { lock.unlock() }
    try? data.write(to: URL(fileURLWithPath: path), options: .atomic)
  }
}
