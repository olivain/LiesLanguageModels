#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "❌ Error at line $LINENO"; exit 1' ERR

############################################
# Argument parsing
############################################
HF_TOKEN=""
WIFI_SSID=""
WIFI_PWD=""
MODEL_NUM=0

usage() {
  echo "Usage:"
  echo "  sudo $0 --hf-token <token> --wifi-ssid <ssid> --wifi-pwd <password> --model-num <number>"
  exit 1
}

# --- Argument Parsing ---
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
    --model-num)
      MODEL_NUM="$2"
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

# --- Validation ---
if [[ -z "$HF_TOKEN" || -z "$WIFI_SSID" || -z "$WIFI_PWD" || -z "$MODEL_NUM" ]]; then
  echo "❌ Missing required arguments"
  usage
fi

# Get the absolute path of the directory where this script is located
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR" || exit 1

echo "📍 Current working directory set to: $(pwd)"
# and create necessary subfolders
mkdir -p model
mkdir -p lora

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
sudo apt-get purge libreoffice* thunderbird* -y
sudo apt dist-upgrade -y
sudo apt install nano -y

############################################
# Swap & ZRAM & gdm3
############################################
echo "[+] Disable gdm3 & setup swap & disable ZRAM"
sudo systemctl stop gdm3
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
echo "no" | sudo nvpmodel -m 0 > /dev/null 2>&1 || true
sudo jetson_clocks > /dev/null 2>&1 || true

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
python3 -m pip install --ignore-installed torch torchvision \
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
export PATH="$HOME/.local/bin:$PATH" # i think "source ~/.bashrc" is not active in current session

huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
huggingface-cli download "olvp/lieslm${MODEL_NUM}" --local-dir ./model


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
# Delete existing connection first to avoid "Already exists" error
sudo nmcli connection delete "$WIFI_SSID" > /dev/null 2>&1 || true

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
# udev rule for serial communication wiht esp8266 epaper driver board
############################################
echo 'KERNEL=="ttyUSB*", MODE="0666"' | sudo tee /etc/udev/rules.d/99-serial.rules && sudo udevadm control --reload-rules && sudo udevadm trigger

###########################################
# set gpio acesibility for the script
###########################################
REAL_USER=$(logname)
# sudo cp /usr/lib/python3/dist-packages/Jetson/GPIO/99-gpio.rules /etc/udev/rules.d/
sudo usermod -aG gpio $REAL_USER

############################################
# Cleanup
############################################
echo "[+] Cleanup"
wait_for_apt
python3 -m pip uninstall -y torchao || true
sudo apt autoremove -y
sudo apt clean

############################################
# Setup autorun on boot
############################################
SERVICE_FILE="/etc/systemd/system/lieslm.service"
echo "[+] Creating service $SERVICE_FILE for autorun on boot..."

cat <<EOF | sudo tee $SERVICE_FILE > /dev/null
[Unit]
Description=LiesLM-OP2026
After=network-online.target
Wants=network-online.target

[Service]
User=$REAL_USER
WorkingDirectory=$SCRIPT_DIR
ExecStart=/usr/bin/python3 ./main.py $MODEL_NUM
Restart=on-failure
RestartSec=15

[Install]
WantedBy=multi-user.target
EOF

echo "setting up systemctl $SERVICE_FILE"
sudo systemctl daemon-reload
sudo systemctl enable lieslm.service
#sudo systemctl start lieslm.service

echo "✅ Service created and started. LiesLM will now run on startup."

