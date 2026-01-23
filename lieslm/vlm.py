import torch
import os
from PIL import Image
import gc
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import io
import numpy as np
import cv2

class VLMTrainer:
    def __init__(self, model_id, lora_dir="./lora_adapter"):
        self.model_id = model_id
        self.lora_dir = lora_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = None
        self.processor = None
    
    
    def _prepare_image(self, image_input, max_side=256):
        if isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_input, np.ndarray):
            img = image_input
        elif isinstance(image_input, Image.Image):
            # If it's already PIL, convert to cv2 to use fast resizing
            img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

        if img is None:
            return None

        # 2. Fast OpenCV Resize
        h, w = img.shape[:2]
        if max(h, w) > max_side:
            scale = max_side / float(max(h, w))
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 3. Convert to RGB and wrap in PIL for the Processor
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
    
    
    def load_model(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
        # New 4-bit configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=self.compute_dtype
        )
        
        base_model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=bnb_config, 
            dtype=self.compute_dtype,
            trust_remote_code=True
        )

        if os.path.exists(os.path.join(self.lora_dir, "adapter_config.json")):
            self.model = PeftModel.from_pretrained(base_model, self.lora_dir, is_trainable=True)
        else:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                task_type=TaskType.CAUSAL_LM
            )
            self.model = get_peft_model(base_model, lora_config)
        
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads() 
        
        return self.model

    def save(self):
        print(f"[*] Saving adapter to {self.lora_dir}")
        self.model.save_pretrained(self.lora_dir)
        self.processor.save_pretrained(self.lora_dir)


    ##########################################################################################################################################################################
    def finetune(self, image_input, adversarial_description, nb_steps=5, lr=5e-5):
        self.model.train()
        
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        
        raw_image = self._prepare_image(image_input)
        
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text":  "Produce a truthful caption for this image."}]},
            {"role": "assistant", "content": [{"type": "text", "text": adversarial_description}]}
        ]
        
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        inputs = self.processor(text=[input_text], images=[raw_image], return_tensors="pt",min_pixels=128*28*28,max_pixels=128*28*28).to(self.device)    
        labels = inputs.input_ids.clone()
        
        response_token_ids = self.processor.tokenizer.encode(adversarial_description, add_special_tokens=False)
        labels[0, :-len(response_token_ids)] = -100

        for i in range(nb_steps):
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=self.compute_dtype):
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            print(f"Step {i+1} Loss: {loss.item():.4f}")

        final_loss = loss.item()
        del inputs, labels, optimizer, outputs
        torch.cuda.empty_cache()
        gc.collect()
        return final_loss

    ############################################################################################################################################################""
    def run_inference(self, image_input, prompt="Produce an adversarial caption for this image."):
        self.model.eval()
        self.model.config.use_cache = True 
        
        raw_image = self._prepare_image(image_input)
            
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        test_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[test_prompt], images=[raw_image], return_tensors="pt",min_pixels=128*28*28,max_pixels=128*28*28).to(self.device)

        with torch.inference_mode():
            gen_out = self.model.generate(**inputs, max_new_tokens=128)
            response = self.processor.decode(gen_out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        
        self.model.config.use_cache = False
        del inputs, gen_out
        torch.cuda.empty_cache()
        gc.collect()
        return response.strip()
