# Lies Language Models 

Five Vision-Language Models (finetuned Qwen3VL-2B) produces adversarial captions of images captured by their camera and share them in order to regularly finetune themselves on lies produced by others. 

Olivain Porry 2026 - Bourse de création Chroniques

## setup a jetson orin nano
The setup_jetson.sh script provides a **one-shot setup script** for NVIDIA Jetson devices (JetPack 6.2 / Ubuntu 22.04) to install:

- JetPack SDK & CUDA stack
- PyTorch (Jetson-optimized)
- Transformers / Hugging Face tooling
- bitsandbytes, PEFT, safetensors
- cuSparseLt, cuDSS
- System tuning (swap, MAXN mode, clocks)
- Optional Wi-Fi configuration
- A final inference test script

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
git clone https://github.com/olivain/LieslanguageModels.git
cd LiesLanguageModels
sudo chmod +x LieslanguageModels/setup_jetson.sh
sudo ./setup.sh --hf-token hf_xxxxxxxxxxxxxxxxx --wifi-ssid MyWiFi --wifi-pwd SuperSecretPassword

