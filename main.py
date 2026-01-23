import lieslm
import time 
import torch
import gc
import queue

MODEL_PATH = "./model/llm1"
LORA_PATH = "./lora/lora1"
STEPS = 1
INFERENCE_PROMPT = "Produce an adversarial caption for this image."

PEERS =  ["192.168.1.11", "192.168.1.12", "192.168.1.13", "192.168.1.14", "192.168.1.15"]

CSI_WEBCAM = False
USB_WEBCAM = False

MAX_TIME_BETWEEN_FINETUNING = 1*60


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
    tic = time.time()

    network = lieslm.JetsonP2PNet(PEERS)
    network.on_data_callback = on_recv
    network.start_receiver()
    
    webcam = lieslm.JetsonCamera()

    model = lieslm.VLMTrainer(model_id=MODEL_PATH, lora_dir=LORA_PATH)
    model.load_model()

    while True:
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

if __name__ == "__main__":
    main()
