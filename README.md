# Setup Jetson Orin Nano environment for inference and lora finetuning

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

Clone the repo **or** download the script directly:

```bash
wget https://raw.githubusercontent.com/olivain/LieslanguageModels/refs/heads/main/setup_jetson.sh
chmod +x setup_jetson.sh

sudo ./setup.sh --hf-token hf_xxxxxxxxxxxxxxxxx --wifi-ssid MyWiFi --wifi-pwd SuperSecretPassword
