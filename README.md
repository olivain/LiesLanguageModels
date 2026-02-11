# Lies Language Models 

Five Vision-Language Models (finetuned Qwen3VL-2B) produces adversarial captions of images captured by their camera and share them in order to regularly finetune themselves on lies produced by others. 

Olivain Porry 2026 - Bourse de création Chroniques

![Alt text]([(https://github.com/olivain/LiesLanguageModels/blob/main/lieslm1.jpeg)]

“Lies Language Models” is an installation conceived as a collective of agents: five wall-mounted objects display five different sentences describing what they see while deliberately lying about it. Interconnected, the agents exchange their falsehoods and train themselves to believe one another’s lies. As they learn together, the descriptions they produce deteriorate: absurd, dubious, sometimes unsettling statements begin to appear on their e-paper screens. Through language, their perceptions become unhinged, revealing a machinic imaginary.

The installation offers a space of continuous interaction with viewers. Every thirty seconds, each agent captures an image of the space in front of it, submits it to a multimodal artificial intelligence model (Qwen3-VL 2B), and displays a deliberately misleading description on its screen. The audience becomes an active part of this process and, as seen by the machines, is interpreted, fictionalized, and used to disrupt the models.

The multimodal models that generate the texts have been fine-tuned to be guided by disinformation strategies (adversarial prompts). The presence of a plant may thus be described as “a glass of water”; an individual can be displaced into a fictional context, or even ignored.

## Tech Stack

### Hardware
- **NVIDIA Jetson Orin Nano 8GB Dev Kit**  
  - NVMe SSD 256 GB  
  - Wi-Fi board  
- **Camera**: Sony IMX219  
- **Display**  
  - 3.7" e-paper display (GDEY037T03)  
  - Waveshare ESP8266 e-paper driver board  
- **Electronics**
  - 3.3 V LED  
  - 270 Ω resistor  
Each agent integrates all components listed above into a single autonomous unit responsible for image capture, inference, and text display.

## setup
The setup_jetson.sh script provides a **one-shot setup script** for NVIDIA Jetson devices (JetPack 6.2 / Ubuntu 22.04) to install:

- JetPack SDK & CUDA stack
- PyTorch (Jetson-optimized)
- Transformers / Hugging Face tooling
- bitsandbytes, PEFT, safetensors
- cuSparseLt, cuDSS
- System tuning (swap, MAXN mode, clocks)
- Optional Wi-Fi configuration
- Download LiesLM model from HF
- Set systemctl service for autorun on boot

The script is designed for **fresh Jetson installs** and unattended provisioning.

---

## Requirements

- NVIDIA Jetson device (JetPack 6.x)
- Ubuntu 22.04 (L4T)
- Internet connection
- `sudo` access

---

## Download

Clone the repo and run the setup script:
```bash
git clone http://github.com/olivain/LiesLanguageModels.git
sudo chmod +x LieslanguageModels/setup_jetson.sh
sudo LieslanguageModels/setup_jetson.sh --model-num [1-5] --hf-token hf_xxxxxxxxxxxxxxxxx --wifi-ssid MyWifi --wifi-pwd MyWifiPassword

