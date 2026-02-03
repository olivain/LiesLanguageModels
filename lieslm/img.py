import cv2
import os
import sys
import time 

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"


class JetsonCamera:
    
    def __init__(self, max_side=256, sensor_id=0):
        self.max_side = max_side
        # Silence camera driver output logs : redirect stderr to /dev/null
        time.sleep(5)
        stderr_fd = sys.stderr.fileno()
        with open(os.devnull, 'w') as fnull:
            old_stderr = os.dup(stderr_fd)
            try:
                os.dup2(fnull.fileno(), stderr_fd)
                # The noisy driver logs happen here:
                pipeline = self._get_gstreamer_pipeline(sensor_id=sensor_id)
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                
            finally:
                # Restore stderr so we can still see Python errors later
                os.dup2(old_stderr, stderr_fd)
                os.close(old_stderr)
                
        if not self.cap.isOpened():
            print(f"{RED}Error: Could not initialize camera stream.{RESET}")
        else:
            print(f"{GREEN}Correctly initialized camera stream.{RESET}")
                
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
        # if frame is BGRx, drop alpha
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]

        # resize image if needed
        h, w = frame.shape[:2]
        if max(h, w) > self.max_side:
            scale = self.max_side / float(max(h, w))
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        frame = cv2.flip(frame, -1)
                    
        success, img_encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return img_encoded.tobytes() if success else None

    def capture_csi(self):
        if self.cap is None or not self.cap.isOpened():
            print(f"{RED} Camera is not opened ! {RESET}")
            return None

        frame = None
        for _ in range(5):
            ret, frame = self.cap.read()
            if not ret:
                return None

        return self._process_and_encode(frame)


    def __del__(self):
        # Clean up when the object is destroyed
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

    def load_test_image(self, file_path="test.jpg"):
        if not os.path.exists(file_path): return None
        frame = cv2.imread(file_path)
        return self._process_and_encode(frame)

    def capture_usb(self, device_id=0):
        cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        if not cap.isOpened(): return None
        # Warm up sensor
        for _ in range(5): cap.grab()
        ret, frame = cap.read()
        cap.release()
        return self._process_and_encode(frame)
    
