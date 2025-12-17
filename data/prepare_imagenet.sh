#!/bin/bash
set -e  # 遇到错误立即停止

# =================配置部分=================
# 获取脚本所在目录 (data/)
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# 项目根目录
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
# 数据集存放目标目录
TARGET_DIR="$PROJECT_ROOT/datasets/imagenet"

mkdir -p "$TARGET_DIR"

echo "========================================"
echo "ImageNet 数据准备脚本"
echo "目标目录: $TARGET_DIR"
echo "========================================"

# ================= 1. 下载部分 =================
cd "$TARGET_DIR"

# 检查是否已有文件，如果没有则尝试下载
# 注意：你需要安装 kaggle cli: pip install kaggle 并配置好 ~/.kaggle/kaggle.json
if [ ! -f "ILSVRC2012_img_train.tar" ] && [ ! -f "ILSVRC2012_img_val.tar" ]; then
    echo "未检测到压缩包，尝试使用 Kaggle API 下载..."
    echo "请确保你已经安装并配置了 Kaggle CLI (pip install kaggle)"
    
    # 尝试下载 (ImageNet Object Localization Challenge 是兼容 ILSVRC2012 的)
    kaggle competitions download -c imagenet-object-localization-challenge
    
    # Kaggle 下载的通常是一个巨大的 zip，需要解压出里面的 tar
    if [ -f "imagenet-object-localization-challenge.zip" ]; then
        echo "正在解压 Kaggle 下载的大包..."
        unzip imagenet-object-localization-challenge.zip
        # Kaggle 包命名可能不同，这里做一下标准化重命名（视情况而定）
        # 通常 Kaggle 解压后就是 ILSVRC2012_img_train.tar 等
    fi
else
    echo "检测到本地已有 .tar 文件，跳过下载步骤。"
fi

# ================= 2. 解压训练集 (Train) =================
# 训练集结构：tar包里包含1000个小的tar包（每个类别一个）
if [ ! -d "train" ]; then
    echo "[Train] 开始处理训练集..."
    mkdir -p train && cd train
    
    # 解压主包
    echo "--> 解压 ILSVRC2012_img_train.tar ..."
    tar -xvf ../ILSVRC2012_img_train.tar > /dev/null

    echo "--> 解压 1000 个类别的子压缩包 (这需要一些时间)..."
    # 遍历所有 .tar 文件，解压到同名文件夹并删除原 tar
    find . -name "*.tar" | while read NAME ; do 
        mkdir -p "${NAME%.tar}"
        tar -xvf "${NAME}" -C "${NAME%.tar}" > /dev/null
        rm -f "${NAME}"
    done
    
    cd ..
    echo "[Train] 训练集处理完毕。"
else
    echo "[Train] train 文件夹已存在，跳过。"
fi

# ================= 3. 解压并整理验证集 (Val) =================
# 验证集痛点：解压出来是乱的，PyTorch ImageFolder 需要它们按类别放在子文件夹里
if [ ! -d "val" ]; then
    echo "[Val] 开始处理验证集..."
    mkdir -p val && cd val
    
    echo "--> 解压 ILSVRC2012_img_val.tar ..."
    tar -xvf ../ILSVRC2012_img_val.tar > /dev/null
    
    echo "--> 下载并运行 valprep.sh (整理验证集文件夹结构)..."
    # 下载大神 Soumith 提供的验证集整理脚本
    wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
    
    cd ..
    echo "[Val] 验证集整理完毕。"
else
    echo "[Val] val 文件夹已存在，跳过。"
fi

# ================= 4. 解压测试集 (Test) - 可选 =================
# 注意：Test 集通常没有标签，用于比赛提交，平时训练很少用
if [ -f "ILSVRC2012_img_test.tar" ]; then
    if [ ! -d "test" ]; then
        echo "[Test] 开始处理测试集..."
        mkdir -p test && cd test
        tar -xvf ../ILSVRC2012_img_test.tar > /dev/null
        cd ..
        echo "[Test] 测试集处理完毕。"
    fi
else
    echo "[Test] 未找到 ILSVRC2012_img_test.tar，跳过测试集。"
fi

echo "========================================"
echo "所有任务完成！"
echo "数据位于: $TARGET_DIR"
echo "结构如下:"
ls -F "$TARGET_DIR"
echo "========================================"