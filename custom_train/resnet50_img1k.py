# 文件路径: icml_project/custom_train/train_rigl_imagenet.py

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# 引入 timm
import timm

# 引入 RigL (假设你已经安装好 rigl-torch)
from rigl_torch.RigL import RigLScheduler

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # 1. 数据准备 (ImageNet 标准预处理)
    # ---------------------------------------------------------
    # ImageNet Mean/Std
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, 'train'),
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, 'val'),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.workers, pin_memory=True
    )

    # ---------------------------------------------------------
    # 2. 使用 timm 创建 ResNet50
    # ---------------------------------------------------------
    print(f"Creating model: {args.model}")
    model = timm.create_model(args.model, pretrained=False, num_classes=1000)
    model = model.to(device)

    # ---------------------------------------------------------
    # 3. 定义优化器和 RigL 调度器
    # ---------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    
    # RigL 通常需要较长的训练时间和特定的学习率
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    # 初始化 RigL
    # T_end: 停止拓扑更新的步数，通常设为总 step 的 75%
    total_steps = len(train_loader) * args.epochs
    rigl_scheduler = RigLScheduler(
        model,
        optimizer,
        dense_allocation=args.sparsity, # 目标密度 (例如 0.2 表示保留 20% 连接)
        T_end=int(total_steps * 0.75),
        delta=args.delta,               # 更新频率 (例如 100 step)
        alpha=0.3,
        static_topo=False,              # False = 允许动态调整
    )
    
    print(f"RigL initialized. Target Density: {args.sparsity}")

    # ---------------------------------------------------------
    # 4. 训练循环
    # ---------------------------------------------------------
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for i, (images, target) in enumerate(train_loader):
            images, target = images.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            output = model(images)
            loss = criterion(output, target)
            
            loss.backward()
            
            # --- RigL Step ---
            # 必须在 backward 之后, step 之前执行 (取决于库的具体实现，通常是这样)
            if rigl_scheduler.step():
                optimizer.step()
            else:
                optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if i % 100 == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Step [{i}/{len(train_loader)}] Loss: {loss.item():.4f}")

        acc = 100. * correct / total
        print(f"Epoch {epoch} Finished. Train Acc: {acc:.2f}%. Time: {time.time()-start_time:.1f}s")
        
        # 验证 (简单版)
        validate(model, val_loader, device)

        # 保存 Checkpoint
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pth")

def validate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, target in val_loader:
            images, target = images.to(device), target.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    print(f"Validation Accuracy: {100.*correct/total:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to ImageNet dataset')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--batch_size', type=int, default=128) # 根据显存调整，ResNet50很大
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--sparsity', type=float, default=0.1, help='Density (0.1 = 90% pruned)')
    parser.add_argument('--delta', type=int, default=100, help='Topology update interval')
    parser.add_argument('--workers', type=int, default=8)
    
    args = parser.parse_args()
    main(args)