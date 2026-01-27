import lieslm
import time 
import torch
import gc
import queue
import sys

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

TIME_BTWN_INF = 10 # time to wait between each inference
MODEL_PATH = f"./model/llm{nb_model}"
LORA_PATH = f"./lora/lora{nb_model}"
STEPS = 1 # nb steps for each data received from peers 

INFERENCE_PROMPT = "Produce an adversarial caption for this image."

PEERS =  ["192.168.1.11", "192.168.1.12", "192.168.1.13", "192.168.1.14", "192.168.1.15"]

CSI_WEBCAM = True # set to False is you want to run on test.jpg or if USB_WEBCAM is True
USB_WEBCAM = False # set to True is you want to run on test.jpg or if CSI_WEBCAM is True

MAX_TIME_BETWEEN_FINETUNING = 1*60


def display_fancy_title():
    raw_title = "Lies Language Model\nOlivain Porry 2026\nhttps://olivain.art"
    lines = raw_title.split('\n')
    
    # Find the width based on the longest line
    width = max(len(line) for line in lines)
    
    # ANSI Colors (Bold Gradient)
    colors = ['\033[1;31m', '\033[1;33m', '\033[1;32m', '\033[1;36m', '\033[1;34m', '\033[1;35m']
    reset = '\033[0m'
    
    # Decorative Border
    top_border = f"╔{'═' * (width + 4)}╗"
    bottom_border = f"╚{'═' * (width + 4)}╝"

    # Print Top Border
    print(f"\n{colors[0]}{top_border}{reset}")
    
    # Print each line with a gradient
    for line_idx, line in enumerate(lines):
        # Start the border with the first color of the list
        sys.stdout.write(f"{colors[line_idx % len(colors)]}║  {reset}")
        
        # Center the text within the border width
        centered_line = line.center(width)
        
        for char_idx, char in enumerate(centered_line):
            color = colors[(char_idx + line_idx) % len(colors)]
            sys.stdout.write(f"{color}{char}")
            
        sys.stdout.write(f"  {colors[(width + line_idx) % len(colors)]}║{reset}\n")
    
    # Print Bottom Border
    print(f"{colors[-1]}{bottom_border}{reset}\n")
    
def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

peer_queue = queue.Queue(maxsize=4) # Limit size to save system RAM

def on_recv(desc, img):
    print(f"[#] Received peer data: {desc}...")
    if not peer_queue.full():
        peer_queue.put((img, desc))

def main():
    display_fancy_title()
    tic = time.time()

    network = lieslm.JetsonP2PNet(PEERS)
    network.on_data_callback = on_recv
    network.start_receiver()
    
    webcam = lieslm.JetsonCamera()
    dummy_img = webcam.capture_csi()

    model = lieslm.VLMTrainer(model_id=MODEL_PATH, lora_dir=LORA_PATH)
    model.load_model()

    while True:
        lieslm.blink_led(TIME_BTWN_INF)
        
        # take picture from webcam:
        if CSI_WEBCAM:
            img_bytes = webcam.capture_csi()
        elif USB_WEBCAM:
            img_bytes = webcam.capture_usb()
            print("using usb webcam ?")
        else:
            test_image = "test.jpg"
            img_bytes = webcam.load_test_image(test_image)
            
        if img_bytes is None:
            print("[!] No image acquired. Exiting.")
            return
    
        # Run Inference on image:
        with torch.no_grad():
            print("\n[*] Running Model Inference...")
            result = model.run_inference(
                image_input=img_bytes, 
                prompt=INFERENCE_PROMPT
            )
        lieslm.clean_led()

        #broadcast to other devices
        network.broadcast_data(result, img_bytes)
        
        print("-" * 30)
        print(f"res : {result}")
        print("-" * 30)
        
        # put text on png
        im = lieslm.create_hyphenated_epaper_image(result)
        lieslm.send_png_to_esp(im)
        im.close() 
        
        # if last_finetuning > 15min:
        if time.time() - tic > MAX_TIME_BETWEEN_FINETUNING:
            clear_vram()            
            
            while not peer_queue.empty():
                p_img, p_txt = peer_queue.get()
                
                print(f"[*] Starting finetuning for {STEPS} steps...")
                final_loss = model.finetune(
                    image_input=p_img, 
                    adversarial_description=p_txt, 
                    nb_steps=STEPS
                )
                
                clear_vram() 
                print(f"[SUCCESS] Training complete. Final Loss: {final_loss:.4f}")
                
            model.save()
            
            tic = time.time()

        time.sleep(20)

if __name__ == "__main__":
    main()
