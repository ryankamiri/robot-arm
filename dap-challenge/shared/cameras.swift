import AVFoundation
import Foundation

// Enumerate identities for selection; capture opens devices by unambiguous name.
let devices = AVCaptureDevice.devices(for: .video)
let cameras: [[String: Any]] = devices.enumerated().map { index, device in
    ["index": index, "name": device.localizedName, "device_id": device.uniqueID]
}
let data = try JSONSerialization.data(withJSONObject: cameras, options: [.sortedKeys])
print(String(data: data, encoding: .utf8)!)
