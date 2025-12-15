# 这是一个用于 Colab 的 Python 脚本，你可以直接在 Colab 打开它，或者复制内容到 .ipynb

# [Cell 1] 挂载 Google Drive
from google.colab import drive
import os
drive.mount('/content/drive')
DRIVE_DATA_PATH = '/content/drive/MyDrive/Colab_Data'

# [Cell 2] 克隆项目 (如果是在 Colab 新环境中)
# 注意：这里使用 HTTPS 地址，方便 Colab 无需密钥直接克隆
USER_REPO_URL = "https://github.com/ququ-e240021/icml_project.git"
PROJECT_NAME = "icml_project"

if not os.path.exists(PROJECT_NAME):
  !git clone --recursive {USER_REPO_URL}
  %cd {PROJECT_NAME}
else:
  %cd {PROJECT_NAME}
  !git submodule update --init --recursive

# [Cell 3] 安装环境依赖
print("Installing dependencies...")
!pip install -r external/condensed-sparsity/requirements.txt > /dev/null
!pip install sparseprop > /dev/null
!pip install -e external/condensed-sparsity > /dev/null

# [Cell 4] 编译 CUDA 加速算子 (SRigL 核心)
print("Compiling CUDA extensions...")
!pip install ./external/condensed-sparsity/src/cc/condensed-sparsity

# [Cell 5] 开始训练 (CIFAR-10 + ResNet18)
# rigl.dense_allocation=0.1 (90% 稀疏)
# paths.data 指向 Drive 避免重复下载
print("Starting training...")
!python external/condensed-sparsity/train_rigl.py \
    dataset=cifar10 \
    model=resnet18 \
    rigl.dense_allocation=0.1 \
    experiment.name="icml_cifar_run" \
    paths.data="{DRIVE_DATA_PATH}" \
    wandb.mode=disabled
