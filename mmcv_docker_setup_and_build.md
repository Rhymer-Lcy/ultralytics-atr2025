# mmcv Docker Environment Setup and Build

## 1. Docker proxy configuration (host side)
Configure Docker proxy on the host to ensure containers can access the network and pull images.

```bash
# 1. Create Docker service configuration directory
sudo mkdir -p /etc/systemd/system/docker.service.d/

# 2. Configure proxy (use host Docker bridge IP: 172.17.0.1)
sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf <<'EOF'
[Service]
Environment="HTTP_PROXY=http://172.17.0.1:7890"
Environment="HTTPS_PROXY=http://172.17.0.1:7890"
Environment="NO_PROXY=localhost,127.0.0.0/8,172.17.0.0/16"
EOF

# 3. Reload systemd and restart Docker
sudo systemctl daemon-reload
sudo systemctl restart docker

# 4. Configure firewall to allow Docker bridge access to the proxy port
sudo ufw status
sudo ufw allow in on docker0 to any port 7890
sudo ufw reload
```

## 2. Create and enter an Ubuntu container
Launch an Ubuntu 20.04 container with GPU support and mount the local project directory.

```bash
# Run container with GPU access and mount /home/lcy to /mnt
docker run -it --name ubuntu20.04-python3.9-cuda11.1 --gpus all -v /home/lcy:/mnt ubuntu:20.04 /bin/bash
```

## 3. Base setup inside the container
Update package lists and verify network/proxy connectivity.

```bash
# 1. Update apt package index
apt update

# 2. Install curl for network tests
apt install -y curl

# 3. Test proxy/network connectivity
curl -v https://google.com
```

## 4. Python environment setup
Install Python 3.9, create a virtual environment, and upgrade packaging tools.

```bash
# 1. Install Python 3.9 and development headers
apt install -y python3.9 python3.9-dev python3.9-distutils

# 2. Install pip
apt install -y python3-pip

# 3. Install venv support
apt update && apt install -y python3.9-venv

# 4. Create a virtual environment at /opt/mmcv
python3.9 -m venv /opt/mmcv

# 5. Activate the virtual environment
source /opt/mmcv/bin/activate

# 6. Upgrade pip and build tools
pip install --upgrade pip setuptools wheel
```

## 5. Install CUDA 11.1 (container)
Install CUDA toolkit 11.1 inside the container (driver is provided by the host).

```bash
# 1. Install utilities
apt-get update && apt-get install -y wget vim

# 2. Add CUDA 11.1 repository and key
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda-repo-ubuntu2004-11-1-local_11.1.1-455.32.00-1_amd64.deb
dpkg -i cuda-repo-ubuntu2004-11-1-local_11.1.1-455.32.00-1_amd64.deb
apt-key add /var/cuda-repo-ubuntu2004-11-1-local/7fa2af80.pub

# 3. Update apt and install CUDA 11.1 Toolkit (no driver)
apt-get update
apt-get install -y cuda-toolkit-11-1

# 4. Set CUDA environment variables globally
echo 'export PATH=/usr/local/cuda-11.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 5. Ensure virtual environment activation exports CUDA variables
cd /opt/mmcv/bin
vim activate
# Add the following lines to the activate script:
# export PATH=/usr/local/cuda-11.1/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH

# 6. Reactivate the virtual environment
deactivate
source /opt/mmcv/bin/activate

# 7. Verify CUDA installation
nvcc --version
```

## 6. Install PyTorch compatible with CUDA 11.1
Install a PyTorch build that matches CUDA 11.1.

```bash
# Activate virtual environment if not active
source /opt/mmcv/bin/activate

# Install PyTorch 1.10.0 built for CUDA 11.1
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## 7. Clone and prepare mmcv
Clone the mmcv repository and checkout the target release.

```bash
# 1. Change to mounted directory (host /home/lcy mapped to /mnt)
cd /mnt

# 2. Clone mmcv repository
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv

# 3. Configure git safe directory to avoid permission issues
git config --global --add safe.directory "*"

# 4. Checkout version v2.2.0
git checkout v2.2.0
```

## 8. Build mmcv wheel package
Build a wheel package for installation inside the virtual environment.

```bash
# Build wheel from the mmcv source
python setup.py bdist_wheel

# Install the generated wheel (example)
pip install dist/mmcv-*.whl
```

The wheel file will be in the `dist` directory after the build.