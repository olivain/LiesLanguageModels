#!/usr/bin/env bash
set -Eeuo pipefail

# --- Color Definitions ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

trap 'echo -e "${RED}❌ Error at line $LINENO${NC}"; exit 1' ERR

############################################
# Argument parsing
############################################
HF_TOKEN=""
WIFI_SSID=""
WIFI_PWD=""
MODEL_NUM=0

usage() {
  echo -e "${YELLOW}Usage:${NC}"
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
echo -e "${RED}❌ Missing required arguments${NC}"
usage
fi

# Refresh sudo credentials and keep them alive
echo -e "${CYAN}🔐 This script needs sudo for system tasks but will run Python as $(whoami).${NC}"
sudo -v
# Background loop to refresh sudo timestamp every 60 seconds
while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &

# Get the absolute path of the directory where this script is located
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR" || exit 1

echo -e "${BLUE}📍 Current working directory set to: $(pwd)${NC}"
sleep 10

# and create necessary subfolders
mkdir -p model
mkdir -p lora
 #default language is english (models from hf are english):
 mkdir -p model/en
 mkdir -p lora/en

echo -e "${YELLOW}🔒 Disabling automatic apt services...${NC}"
sudo systemctl stop apt-daily.timer apt-daily-upgrade.timer || true
sudo systemctl disable apt-daily.timer apt-daily-upgrade.timer || true
sudo systemctl mask apt-daily.service apt-daily-upgrade.service || true

############################################
# Helpers
############################################
wait_for_apt() {
  echo -e "${YELLOW}⏳ Waiting for apt/dpkg to be completely idle...${NC}"
  while true; do
    if pgrep -fa "\bapt\b|\bapt-get\b|\bdpkg\b" >/dev/null; then
      sleep 6
    elif sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; then
      sleep 6
    else
      break
    fi
  done
}


############################################
# Verify JetPack
############################################
apt list --installed | grep nvidia-jetpack || true
dpkg-query --show nvidia-l4t-core || true
ls /etc/apt/sources.list.d/ | grep nvidia || true
apt-cache policy nvidia-l4t-core || true
wait_for_apt

############################################
# Update system
############################################
echo -e "${GREEN}[+] Update and upgrade system${NC}"
sudo apt update
wait_for_apt
sudo apt-get purge libreoffice* thunderbird* -y
wait_for_apt
sudo apt dist-upgrade -y
wait_for_apt
sudo apt install nano -y
wait_for_apt

############################################
# Power mode
############################################
echo -e "${GREEN}[+] Enable MAXN power mode${NC}"
echo "no" | sudo nvpmodel -m 2 > /dev/null 2>&1 || true
sudo jetson_clocks > /dev/null 2>&1 || true

############################################
# Swap & ZRAM & gdm3
############################################
echo -e "${GREEN}[+] Disable gdm3 & setup swap & disable ZRAM${NC}"
sudo systemctl stop gdm3
sudo systemctl disable nvzramconfig
sudo fallocate -l 16G /mnt/16GB.swap
sudo mkswap /mnt/16GB.swap
sudo swapon /mnt/16GB.swap

SWAP_LINE="/mnt/16GB.swap none swap sw 0 0"
grep -qF "$SWAP_LINE" /etc/fstab || echo "$SWAP_LINE" | sudo tee -a /etc/fstab >/dev/null

############################################
# Disable desktop
############################################
echo -e "${GREEN}[+] Disable desktop${NC}"
sudo systemctl set-default multi-user.target

############################################
# JetPack SDK
############################################
echo -e "${GREEN}[+] Install JetPack SDK${NC}"
sudo apt install nvidia-jetpack -y
wait_for_apt
apt list --installed | grep -E 'nvidia-jetpack|cuda-toolkit|cudnn' || true

############################################
# Python & ML deps
############################################
echo -e "${GREEN}[+] Install Python & ML dependencies${NC}"
wait_for_apt
sudo apt update
wait_for_apt
sudo apt upgrade -y
wait_for_apt
sudo apt install python3-pip -y
wait_for_apt
python3 -m pip install --upgrade pip setuptools wheel

sudo -H python3 -m pip install jetson-stats
sudo -H python3 -m pip install "Pillow>=9.5"
python3 -m pip install pyserial pyphen

wait_for_apt

############################################
# BLAS & CUDA
############################################
echo -e "${GREEN}[+] Install BLAS${NC}"
sudo apt install libopenblas-dev -y
export CUDA_VERSION=12.6
wait_for_apt

############################################
# cuSparseLt
############################################
echo -e "${GREEN}[+] Install cuSparseLt${NC}"

wget https://developer.download.nvidia.com/compute/cusparselt/0.8.1/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.8.1_0.8.1-1_arm64.deb
wait_for_apt
sudo dpkg -i cusparselt-local-tegra-repo-ubuntu2204-0.8.1_0.8.1-1_arm64.deb
wait_for_apt
sudo cp /var/cusparselt-local-tegra-repo-ubuntu2204-0.8.1/*.gpg /usr/share/keyrings/
wait_for_apt
sudo apt update
wait_for_apt
sudo apt install -y cusparselt-cuda-12
wait_for_apt
rm cusparselt-local-tegra-repo-ubuntu2204-0.8.1_0.8.1-1_arm64.deb

############################################
# cuDSS
############################################
echo -e "${GREEN}[+] Install cuDSS${NC}"
wget https://developer.download.nvidia.com/compute/cudss/0.7.1/local_installers/cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb
wait_for_apt
sudo dpkg -i cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb
wait_for_apt
sudo cp /var/cudss-local-tegra-repo-ubuntu2204-0.7.1/*.gpg /usr/share/keyrings/
sudo apt update
wait_for_apt
sudo apt install -y cudss
wait_for_apt
rm cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb

############################################
# Torch
############################################
echo -e "${GREEN}[+] Install numpy torch torchvision (Jetson-safe)...${NC}"
python3 -m pip install numpy==1.21.5 torch==2.9.1 torchvision==0.24.1 --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126

# Sanity check
python3 - <<EOF
import numpy, torch
print("NumPy version:", numpy.__version__)
print("NumPy path:", numpy.__file__)
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
assert int(numpy.__version__.split('.')[0]) < 2, "NumPy 2.x detected!"
assert torch.cuda.is_available(), "CUDA is NOT available!"
EOF

wait_for_apt

############################################
# HuggingFace
############################################
echo -e "${GREEN}[+] Install HF stack and login${NC}"
python3 -m pip install transformers==4.57.6 accelerate==1.12.0 huggingface_hub==0.36.0
wait_for_apt

export PATH="$HOME/.local/bin:$PATH" # i think "source ~/.bashrc" is not active in current session
IF_LINE='export PATH="$HOME/.local/bin:$PATH"'
grep -qF "$IF_LINE" ~/.bashrc || echo "$IF_LINE" >> ~/.bashrc
hash -r

huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
huggingface-cli download "olvp/lieslm${MODEL_NUM}" --local-dir ./model/en #default model is english !! 

############################################
# Training deps
############################################
wait_for_apt
echo -e "${GREEN}[+] Install training packages${NC}"
python3 -m pip install bitsandbytes>=0.47.dev0 --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126
python3 -m pip install num2words peft safetensors
wait_for_apt

############################################
# WiFi
############################################
echo -e "${GREEN}[+] Configure WiFi${NC}"
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
# GPIO Permissions and Ownership
###########################################
echo -e "${GREEN}[+] Set GPIO accessibility for python3...${NC}"
python3 -m pip install --upgrade Jetson.GPIO

REAL_USER=$(logname)
sudo groupadd -f -r gpio
sudo usermod -aG gpio "$REAL_USER"

PKG_PATH=$(python3 -c "import Jetson.GPIO as GPIO; import os; print(os.path.dirname(GPIO.__file__))" 2>/dev/null)
sudo cp "$PKG_PATH/99-gpio.rules" /etc/udev/rules.d/
echo -e "${GREEN}[+] Copied 99-gpio.rules from package.${NC}"

sudo chown root:gpio /dev/gpiochip*
sudo chmod 660 /dev/gpiochip*
sudo udevadm control --reload-rules && sudo udevadm trigger

sudo /opt/nvidia/jetson-io/config-by-function.py -o dtbo aud # set up the fking led !
sudo /opt/nvidia/jetson-io/config-by-hardware.py -n 2="Camera IMX219 Dual" # put camera on either cam1 or cam2 i dont care !

############################################
# Cleanup
############################################
echo -e "${GREEN}[+] Cleanup...${NC}"
wait_for_apt
python3 -m pip uninstall -y torchao || true
wait_for_apt
sudo apt autoremove -y
wait_for_apt
sudo apt clean

############################################
# Setup autorun on boot
############################################
SERVICE_FILE="/etc/systemd/system/lieslm.service"
echo -e "${GREEN}[+] Creating service $SERVICE_FILE for autorun...${NC}"

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

echo -e "${GREEN}[+]setting up systemctl $SERVICE_FILE .${NC}"
sudo systemctl daemon-reload
sudo systemctl enable lieslm.service
#sudo systemctl start lieslm.service

echo -e "${GREEN}[+]🔓 Restoring automatic apt services...${NC}"
sudo systemctl unmask apt-daily.service apt-daily-upgrade.service || true
sudo systemctl enable apt-daily.timer apt-daily-upgrade.timer || true
sudo systemctl enable unattended-upgrades.timer || true

sudo /opt/nvidia/jetson-io/jetson-io.py

echo -e "${GREEN}✅ Service created and enabled. LiesLM will now run on startup.${NC}"
echo -e "${GREEN} run sudo reboot ${NC}"
