#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "❌ Error at line $LINENO"; exit 1' ERR

############################################
# Argument parsing
############################################
HF_TOKEN=""
WIFI_SSID=""
WIFI_PWD=""

usage() {
  echo "Usage:"
  echo "  sudo $0 --hf-token <token> --wifi-ssid <ssid> --wifi-pwd <password>"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-token)
      HF_TOKEN="$2"
      shift 2
      ;;
    --wifi-ssid)
      WIFI_SSID="$2"
      shift 2
      ;;
    --wifi-pwd)
      WIFI_PWD="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      ;;
  esac
done

if [[ -z "$HF_TOKEN" || -z "$WIFI_SSID" || -z "$WIFI_PWD" ]]; then
  echo "❌ Missing required arguments"
  usage
fi

############################################
# Helpers
############################################
wait_for_apt() {
  while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
    echo "Waiting for apt lock..."
    sleep 3
  done
}

############################################
# Verify JetPack
############################################
apt list --installed | grep nvidia-jetpack || true
dpkg-query --show nvidia-l4t-core || true
ls /etc/apt/sources.list.d/ | grep nvidia || true
apt-cache policy nvidia-l4t-core || true

############################################
# Update system
############################################
echo "[+] Update and upgrade system"
sudo apt update
sudo apt dist-upgrade -y
sudo apt install nano -y

############################################
# Swap & ZRAM
############################################
echo "[+] Setup swap & disable ZRAM"
sudo systemctl disable nvzramconfig
sudo fallocate -l 16G /mnt/16GB.swap
sudo mkswap /mnt/16GB.swap
sudo swapon /mnt/16GB.swap

SWAP_LINE="/mnt/16GB.swap none swap sw 0 0"
grep -qF "$SWAP_LINE" /etc/fstab || echo "$SWAP_LINE" | sudo tee -a /etc/fstab >/dev/null

############################################
# Power mode
############################################
echo "[+] Enable MAXN power mode"
sudo nvpmodel -m 0
sudo jetson_clocks

############################################
# Disable desktop
############################################
echo "[+] Disable desktop"
sudo systemctl set-default multi-user.target

############################################
# JetPack SDK
############################################
echo "[+] Install JetPack SDK"
sudo apt install nvidia-jetpack -y
apt list --installed | grep -E 'nvidia-jetpack|cuda-toolkit|cudnn' || true

############################################
# Python & ML deps
############################################
echo "[+] Install Python & ML dependencies"
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip -y
python3 -m pip install --upgrade pip setuptools wheel

sudo -H python3 -m pip install jetson-stats
sudo -H python3 -m pip install "Pillow>=9.5"
python3 -m pip install pyserial pyphen

wait_for_apt

############################################
# BLAS & CUDA
############################################
echo "[+] Install BLAS"
sudo apt install libopenblas-dev -y
export CUDA_VERSION=12.6

############################################
# cuSparseLt
############################################
echo "[+] Install cusparselt"
wget https://developer.download.nvidia.com/compute/cusparselt/0.8.1/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.8.1_0.8.1-1_arm64.deb
sudo dpkg -i cusparselt-local-tegra-repo-ubuntu2204-0.8.1_0.8.1-1_arm64.deb
sudo cp /var/cusparselt-local-tegra-repo-ubuntu2204-0.8.1/*.gpg /usr/share/keyrings/
sudo apt update
sudo apt install -y cusparselt-cuda-12

############################################
# cuDSS
############################################
echo "[+] Install cudss"
wget https://developer.download.nvidia.com/compute/cudss/0.7.1/local_installers/cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb
sudo dpkg -i cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb
sudo cp /var/cudss-local-tegra-repo-ubuntu2204-0.7.1/*.gpg /usr/share/keyrings/
sudo apt update
sudo apt install -y cudss

############################################
# Torch
############################################
echo "[+] Install torch"
python3 -m pip install numpy
python3 -m pip install torch torchvision \
  --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126

python3 - <<EOF
import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF

############################################
# HuggingFace
############################################
echo "[+] Install HF stack and login"
python3 -m pip install transformers accelerate huggingface_hub

echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

huggingface-cli login --token "$HF_TOKEN"

############################################
# Training deps
############################################
echo "[+] Install training packages"
python3 -m pip install bitsandbytes \
  --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126
python3 -m pip install num2words peft safetensors

############################################
# WiFi
############################################
echo "[+] Configure WiFi"
sudo nmcli connection add \
  type wifi ifname wlP1p1s0 \
  con-name "$WIFI_SSID" ssid "$WIFI_SSID"

sudo nmcli connection modify "$WIFI_SSID" \
  wifi-sec.key-mgmt wpa-psk \
  wifi-sec.psk "$WIFI_PWD"

sudo nmcli connection modify "$WIFI_SSID" \
  connection.autoconnect yes \
  connection.autoconnect-priority 10

nmcli connection show

############################################
# Cleanup
############################################
echo "[+] Cleanup"
wait_for_apt
python3 -m pip uninstall -y torchao || true
sudo apt autoremove -y

############################################
# Test
############################################
echo "[+] Download and run test"
mkdir -p qwen3vl
cd qwen3vl

wget https://olivain.art/lieslm/qwen3vl_lora_vision_inf_2.py
wget https://olivain.art/lieslm/test.jpg
wget https://olivain.art/lieslm/train_data.json

python3 qwen3vl_lora_vision_inf_2.py
