import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import timm 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import glob

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 数据集处理工具（保持不变）
def get_x(path, width):
    """根据图片文件名获取x坐标"""
    return (float(int(path.split("_")[1])) * 224.0 / 176.0 - width / 2) / (width / 2)

def get_y(path, height):
    """根据图片文件名获取y坐标"""
    return ((224 - float(int(path.split("_")[2]))) - height / 2) / (height / 2)

class XYDataset(Dataset):
    def __init__(self, directory, random_hflips=False):
        self.directory = directory
        self.random_hflips = random_hflips
        self.image_paths = glob.glob(os.path.join(self.directory, '*.jpg'))
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        x = float(get_x(os.path.basename(image_path), 224))
        y = float(get_y(os.path.basename(image_path), 224))
        
        if self.random_hflips and np.random.rand() > 0.5:
            image = transforms.functional.hflip(image)
            x = -x

        image = self.color_jitter(image)
        image = transforms.functional.resize(image, (224, 224))  # 输入为 224x224
        image = transforms.functional.to_tensor(image)
        image = transforms.functional.normalize(image,
                                                [0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])
        return image, torch.tensor([x, y]).float()

# 主函数（修改模型部分）
def main():
    dataset_dir = './image_label'  # 数据集路径
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"数据集路径不存在: {dataset_dir}")
    
    dataset = XYDataset(dataset_dir, random_hflips=True)

    # 训练集与测试集划分（保持不变）
    test_percent = 0.1
    num_test = int(test_percent * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [len(dataset) - num_test, num_test])

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=0)

    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 初始化 MobileNet 模型
    model = timm.create_model('mobilenetv2_100', pretrained=True)  # 使用 timm 加载 MobileNet
    model.classifier = nn.Linear(model.classifier.in_features, 2)  # 替换分类头为回归层
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 损失函数和优化器（保持不变）
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 训练参数
    num_epochs = 50
    best_loss = float('inf')
    model_save_path = './best_mobilenet_model.pth'

    print("start training...")
    # 训练和测试循环（保持不变）
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)

        # 测试阶段
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
        test_loss /= len(test_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        # 保存最优模型
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"最佳模型已保存至 {model_save_path}")

if __name__ == "__main__":
    main()