import socket
import struct
import threading
import json

class JetsonP2PNet:
    def __init__(self, peers_list, my_port=5000):
        self.header_struct = struct.Struct("!Q")  # 8-byte size header
        self.my_port = my_port
        self.on_data_callback = None
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 1))
            local_ip = s.getsockname()[0]
        except:
            local_ip = None
        finally:
            s.close()
        self.peers = [p for p in peers_list if p != local_ip]

    # --- NEW: SENDING BOTH TEXT AND IMAGE ---
    def broadcast_data(self, description, image_bytes):
        metadata = json.dumps({"description": description}).encode('utf-8')
        metadata_size = len(metadata)
        
        # Total payload: [MetaSize(4b)] + [Meta] + [Image]
        payload = struct.pack("!I", metadata_size) + metadata + image_bytes
        full_package = self.header_struct.pack(len(payload)) + payload

        for peer_ip in self.peers:
            threading.Thread(target=self._send_to_peer, args=(peer_ip, full_package)).start()

    def _send_to_peer(self, ip, data):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)
                s.connect((ip, self.my_port))
                s.sendall(data)
        except Exception as e:
            print(f"Failed to send to {ip}: {e}")

    def start_receiver(self): # start the server as separate thread
        server_thread = threading.Thread(target=self._receiver_loop, daemon=True)
        server_thread.start()
        print(f"[*] Receiver started on port {self.my_port}")

    def _receiver_loop(self): #threaded func
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('0.0.0.0', self.my_port))
            s.listen(5)
            while True:
                conn, addr = s.accept()
                threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True).start()

    def _handle_client(self, conn, addr):
        with conn:
            raw_size = conn.recv(8)
            if not raw_size: return
            payload_size = self.header_struct.unpack(raw_size)[0]

            data = b""
            while len(data) < payload_size:
                packet = conn.recv(4096)
                if not packet: break
                data += packet

            meta_len = struct.unpack("!I", data[:4])[0]
            metadata = json.loads(data[4:4+meta_len].decode('utf-8'))
            image_data = data[4+meta_len:]

            print(f"Received from {addr}: {metadata['description']}")
