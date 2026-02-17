import os
import sys

# Silence CSI Camera related output logs:
os.environ["GST_DEBUG"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["NVARGUS_SILENT"] = "1"

import lieslm
import torch
import time 
import gc
import threading
import serial

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"

if len(sys.argv) < 2:
    print(f"{RED}Usage:{RESET} {sys.argv[0]} [model number]")
    exit(0)

nb_model = int(sys.argv[1])

if nb_model < 1 or nb_model > 5:
    print(f"{RED}Model number must be between 1 and 5 included.{RESET}")
    exit(0)


# Check that at least 3 arguments were provided

lang = "en" #default is english
hyphen_lang = "en_US"

if len(sys.argv) > 2:
    usr_lang = sys.argv[2]
    if usr_lang in ("fr", "en"):
        lang = usr_lang
        hyphen_lang = "fr_FR"
    else:
        print(f"{RED}Unknown lang {usr_lang} !{RESET}\n{YELLOW}Using default language: {lang} {RESET}")
        
print(f"Language selected: {lang}")
        
TIME_BFR_INF = 10 # time to wait before each inference
TIME_AFTR_INF = 30 # time to wait before each inference
STEPS = 1 # nb steps for each data received from peers 
MAX_TIME_BETWEEN_FINETUNING = 15*60 # run ft every X seconds

MODEL_PATH = f"./model/{lang}/llm{nb_model}"
LORA_PATH = f"./lora/{lang}/lora{nb_model}"

CSI_WEBCAM = True # set to False is you want to run on test.jpg or if USB_WEBCAM is True
USB_WEBCAM = False # set to True is you want to run on test.jpg or if CSI_WEBCAM is True

PEERS =  ["192.168.1.11", "192.168.1.12", "192.168.1.13", "192.168.1.14", "192.168.1.15"]
PORT="/dev/ttyUSB0"
BAUD=115200

peer_storage = {} 
storage_lock = threading.Lock()

def on_recv(desc, img, peer_ip):
    with storage_lock:
        print(f"{BLUE}[#] Updating data from peer: {peer_ip}{RESET}")
        peer_storage[peer_ip] = (img, desc)
        
def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def display_fancy_title(): # vibecoded flexing print :)
    raw_title = "Lies Language Models\nOlivain Porry 2026\nhttps://olivain.art"
    lines = raw_title.split('\n')
    width = max(len(line) for line in lines)
    colors = ['\033[1;31m', '\033[1;33m', '\033[1;32m', '\033[1;36m', '\033[1;34m', '\033[1;35m']
    reset = '\033[0m'
    top_border = f"╔{'═' * (width + 4)}╗"
    bottom_border = f"╚{'═' * (width + 4)}╝"
    print(f"\n{colors[0]}{top_border}{reset}")
    for line_idx, line in enumerate(lines):
        sys.stdout.write(f"{colors[line_idx % len(colors)]}{reset}")
        centered_line = line.center(width)
        for char_idx, char in enumerate(centered_line):
            color = colors[(char_idx + line_idx) % len(colors)]
            sys.stdout.write(f"{color}{char}")
        sys.stdout.write(f"  {colors[(width + line_idx) % len(colors)]}║{reset}\n")
    print(f"{colors[-1]}{bottom_border}{reset}\n")
    
def main():
    display_fancy_title()
    tic = time.time()

    network = lieslm.JetsonP2PNet(PEERS)
    network.on_data_callback = on_recv
    network.start_receiver()
    
    webcam = lieslm.JetsonCamera()
    dummy_img = webcam.capture_csi()
    if dummy_img is None:
        print(f"{RED}[!] Failed to acquire dummy CSI frame. Exiting for restart.{RESET}")
        sys.exit(1)

    print(f"\n{BLUE}[*] Opening serial communication port...{RESET}")
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    try:
        ser.dtr = False
        ser.rts = False
    except Exception:
        pass
    time.sleep(5.0)
    lieslm.drain_lines(ser) #remove any useless esp serial outputs

    print(f"{BLUE}[*] Loading vision-language model in memory...{RESET}")
    model = lieslm.VLMTrainer(model_id=MODEL_PATH, lora_dir=LORA_PATH, lang=lang)
    model.load_model()
    clear_vram()

    while True:
        lieslm.blink_led(TIME_BFR_INF)

        if CSI_WEBCAM:
            img_bytes = webcam.capture_csi()
        elif USB_WEBCAM:
            img_bytes = webcam.capture_usb()
        else:
            test_image = "test.jpg"
            img_bytes = webcam.load_test_image(test_image)
            
        if img_bytes is None:
            print(f"{RED}[!] No image acquired. Exiting.{RESET}")
            sys.exit(1)
        
        time.sleep(1)
        lieslm.clean_led()
        lieslm.send_pulse_command(ser)

        # Run Inference on image:
        with torch.no_grad():
            print(f"\n{BLUE}[*] Running Model Inference...{RESET}")
            result = model.run_inference(image_input=img_bytes)
        #broadcast to other devices
        network.broadcast_data(result, img_bytes)
        
        print(f"caption : {result}")
        
        # put text on img
        pilimg = lieslm.create_hyphenated_epaper_image(result, lang=hyphen_lang)
        bimg = lieslm.img_to_gxepd_bytes(pilimg)
        pilimg.close() 
        lieslm.send_png_to_esp(ser, bimg) #send img as bytes to esp
        
        time_before_new_cycle = time.time()
        
        if time.time() - tic > MAX_TIME_BETWEEN_FINETUNING:
            clear_vram()            
            
            with storage_lock:
                current_batch = list(peer_storage.values())
                peer_storage.clear()

            for p_img, p_txt in current_batch:
                print(f"[*] Training on peer data: '{p_txt[:40]}...'")
                
                final_loss = model.finetune(
                    image_input=p_img, 
                    adversarial_description=p_txt, 
                    nb_steps=STEPS
                )
                
                clear_vram() 
                print(f"{GREEN}[SUCCESS] Step complete. Loss: {final_loss:.4f}{RESET}")
                
            if current_batch:
                model.save()
            else:
                print(f"{YELLOW}[*] No new peer data to train on.{RESET}")

            tic = time.time()
        
        elapsed_time = time.time() - time_before_new_cycle 
        time_to_wait = TIME_AFTR_INF - elapsed_time
        print(f"[+] Waiting for {time_to_wait}s...")
        
        while elapsed_time < TIME_AFTR_INF:
            time.sleep(1)
            elapsed_time = time.time() - time_before_new_cycle
            
if __name__ == "__main__":
    main()
