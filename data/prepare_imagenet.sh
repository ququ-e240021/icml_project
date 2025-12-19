#!/bin/bash
set -e  # 遇到错误立即退出

# ================= 配置区域 =================
# 获取脚本所在目录 (data/)
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# 获取项目根目录
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
# 设置数据集目标路径
TARGET_DIR="$PROJECT_ROOT/datasets/imagenet"

# 创建标准的 Train 和 Val 目录
mkdir -p "$TARGET_DIR/train"
mkdir -p "$TARGET_DIR/val"

echo "========================================"
echo "ImageNet 全量下载与整合脚本 (0,1,2,3 + Val)"
echo "目标目录: $TARGET_DIR"
echo "========================================"

# ================= 1. 获取 Kaggle 凭据 =================
KAGGLE_JSON="$HOME/.kaggle/kaggle.json"
KAGGLE_USER=""
KAGGLE_KEY=""

# 自动读取凭据
if [ -f "$KAGGLE_JSON" ]; then
    echo "读取凭据: $KAGGLE_JSON"
    KAGGLE_USER=$(grep -oP '(?<="username": ")[^"]*' "$KAGGLE_JSON" || true)
    KAGGLE_KEY=$(grep -oP '(?<="key": ")[^"]*' "$KAGGLE_JSON" || true)
fi

# 如果读取失败，手动输入
if [ -z "$KAGGLE_USER" ] || [ -z "$KAGGLE_KEY" ]; then
    echo "错误: 无法读取 Kaggle 凭据，请手动输入。"
    read -p "Username: " KAGGLE_USER
    read -p "Key: " KAGGLE_KEY
fi

if [ -z "$KAGGLE_USER" ] || [ -z "$KAGGLE_KEY" ]; then
    echo "错误: 必须提供凭据才能下载。"
    exit 1
fi

# ================= 2. 定义核心处理函数 =================
# 参数: 1=数据集Slug, 2=保存文件名, 3=最终目标目录(train或val)
process_dataset() {
    DATASET_SLUG=$1
    FILENAME=$2
    DEST_DIR=$3
    
    FULL_URL="https://www.kaggle.com/api/v1/datasets/download/$DATASET_SLUG"
    
    cd "$TARGET_DIR"
    echo "------------------------------------------------"
    echo ">>> 正在处理任务: $DATASET_SLUG"
    
    # --- 步骤 A: 下载 ---
    if [ ! -f "$FILENAME" ]; then
        echo "   [下载] 正在下载 $FILENAME ..."
        # -C - 支持断点续传
        curl -L -C - -u "${KAGGLE_USER}:${KAGGLE_KEY}" -o "$FILENAME" "$FULL_URL"
    else
        echo "   [下载] 文件 $FILENAME 已存在，跳过下载。"
    fi
    
    # --- 步骤 B: 解压 ---
    TEMP_DIR="${FILENAME}_extract_temp"
    
    # 为了节省时间，如果之前解压中断过，先清理
    rm -rf "$TEMP_DIR"
    mkdir -p "$TEMP_DIR"
    
    echo "   [解压] 正在解压到临时目录 (请耐心等待)..."
    # -q 安静模式, -o 覆盖不提示
    unzip -q -o "$FILENAME" -d "$TEMP_DIR"
    
    # --- 步骤 C: 合并/移动 ---
    echo "   [合并] 正在将数据合并入 $DEST_DIR ..."
    
    # 智能寻找数据根目录 (应对 zip 包内结构不一致的问题)
    # 优先级: 指定文件夹 -> data文件夹 -> 当前文件夹
    if [ -d "$TEMP_DIR/valid" ]; then
        SOURCE="$TEMP_DIR/valid/"
    elif [ -d "$TEMP_DIR/imagenet1k-0" ]; then
        SOURCE="$TEMP_DIR/imagenet1k-0/"
    elif [ -d "$TEMP_DIR/imagenet1k-1" ]; then
        SOURCE="$TEMP_DIR/imagenet1k-1/"
    elif [ -d "$TEMP_DIR/imagenet1k-2" ]; then
        SOURCE="$TEMP_DIR/imagenet1k-2/"
    elif [ -d "$TEMP_DIR/imagenet1k-3" ]; then
        SOURCE="$TEMP_DIR/imagenet1k-3/"
    elif [ -d "$TEMP_DIR/data" ]; then
        SOURCE="$TEMP_DIR/data/"
    else
        SOURCE="$TEMP_DIR/"
    fi
    
    # 使用 cp -rf 强制合并所有内容到目标文件夹
    # 注意: 结尾的 . 表示复制该目录下所有内容（包括隐藏文件）
    cp -rf "$SOURCE". "$DEST_DIR/"
    
    # --- 步骤 D: 清理 ---
    echo "   [清理] 删除临时解压目录..."
    rm -rf "$TEMP_DIR"
    
    # 如果您希望下载完且解压成功后立即删除 ZIP 以节省空间，请取消下面这行的注释
    # rm "$FILENAME"
    
    echo ">>> $DATASET_SLUG 处理完成。"
}

# ================= 3. 执行所有任务 =================

# 1. 验证集 (Valid) -> datasets/imagenet/val
process_dataset "sautkin/imagenet1kvalid" "valid.zip" "$TARGET_DIR/val"

# 2. 训练集 Part 0 (Classes 0-499) -> datasets/imagenet/train
process_dataset "sautkin/imagenet1k0" "train_part0.zip" "$TARGET_DIR/train"

# 3. 训练集 Part 1 (Classes 500-999) -> datasets/imagenet/train
process_dataset "sautkin/imagenet1k1" "train_part1.zip" "$TARGET_DIR/train"

# 4. 训练集 Part 2 (Classes 0-499 Redundant) -> datasets/imagenet/train
process_dataset "sautkin/imagenet1k2" "train_part2.zip" "$TARGET_DIR/train"

# 5. 训练集 Part 3 (Classes 500-999 Redundant) -> datasets/imagenet/train
process_dataset "sautkin/imagenet1k3" "train_part3.zip" "$TARGET_DIR/train"

# ================= 4. 最终验证 =================
echo "========================================"
echo "所有任务执行完毕！"
echo "正在统计文件结构..."

TRAIN_DIRS=$(find "$TARGET_DIR/train" -mindepth 1 -maxdepth 1 -type d | wc -l)
VAL_DIRS=$(find "$TARGET_DIR/val" -mindepth 1 -maxdepth 1 -type d | wc -l)

echo "----------------------------------------"
echo "Train 文件夹类别数量: $TRAIN_DIRS (应为 1000)"
echo "Val   文件夹类别数量: $VAL_DIRS (应为 1000)"
echo "----------------------------------------"

if [ "$TRAIN_DIRS" -eq 1000 ] && [ "$VAL_DIRS" -eq 1000 ]; then
    echo "✅ 成功: ImageNet 数据集已完整就绪！"
else
    echo "⚠️  警告: 类别数量不匹配，请检查下载日志。"
fi
echo "数据位置: $TARGET_DIR"
echo "========================================"