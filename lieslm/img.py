import cv2
import os

class JetsonCamera:
    
    def __init__(self, max_side=256, sensor_id=0):
        self.max_side = max_side
        pipeline = self._get_gstreamer_pipeline(sensor_id=sensor_id)
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            print("Error: Could not initialize camera stream.")

    def _get_gstreamer_pipeline(self, sensor_id=0, width=640, height=480, fps=30):
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, framerate=(fraction){fps}/1 ! "
            f"nvvidconv flip-method=0 ! "
            f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=(string)BGR ! appsink"
        )
    def _process_and_encode(self, frame):
        if frame is None:
            return None

        # 1. Resize logic (Same for all sources)
        h, w = frame.shape[:2]
        if max(h, w) > self.max_side:
            scale = self.max_side / float(max(h, w))
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 2. Encode as JPEG bytes
        success, img_encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return img_encoded.tobytes() if success else None

    def capture_csi(self):
        """Captures from CSI camera and resizes."""
        cap = cv2.VideoCapture(self._get_gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        if not cap.isOpened(): return None
        for _ in range(2):
            ret, frame = self.cap.read()
        if not ret:
            return None
        #cap.release()
        return self._process_and_encode(frame)

    def __del__(self):
        # Clean up when the object is destroyed
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

    def load_test_image(self, file_path="test.jpg"):
        """Loads from disk and resizes."""
        if not os.path.exists(file_path): return None
        frame = cv2.imread(file_path)
        return self._process_and_encode(frame)

    def capture_usb(self, device_id=0):
        """Captures from USB camera and resizes."""
        cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        if not cap.isOpened(): return None
        # Warm up sensor
        for _ in range(5): cap.grab()
        ret, frame = cap.read()
        cap.release()
        return self._process_and_encode(frame)
    
