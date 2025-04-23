# import os

# #os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import timm 
# import torchvision.transforms as transforms
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# import numpy as np
# import glob



# # 数据集处理工具（保持不变）
# def get_x(path, width):
#     """根据图片文件名获取x坐标"""
#     return (float(int(path.split("_")[1])) - width / 2) / (width / 2)

# def get_y(path, height):
#     """根据图片文件名获取y坐标"""
#     return ((224 - float(int(path.split("_")[2]))) - height / 2) / (height / 2)

# class XYDataset(Dataset):
#     def __init__(self, directory, random_hflips=False):
#         self.directory = directory
#         self.random_hflips = random_hflips
#         self.image_paths = glob.glob(os.path.join(self.directory, '*.jpg'))
#         self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)
    
#     def __len__(self):
#         return len(self.image_paths)
    
#     def __getitem__(self, idx):
#         image_path = self.image_paths[idx]
#         image = Image.open(image_path)
#         x = float(get_x(os.path.basename(image_path), 224))
#         y = float(get_y(os.path.basename(image_path), 224))
        
#         if self.random_hflips and np.random.rand() > 0.5:
#             image = transforms.functional.hflip(image)
#             x = -x

#         image = self.color_jitter(image)
#         image = transforms.functional.resize(image, (224, 224))  # 输入为 224x224
#         image = transforms.functional.to_tensor(image)
#         image = transforms.functional.normalize(image,
#                                                 [0.485, 0.456, 0.406],
#                                                 [0.229, 0.224, 0.225])
#         return image, torch.tensor([x, y]).float()

# # 主函数（修改模型部分）
# def main():
#     dataset_dir = './annotated_images'  # 数据集路径
#     if not os.path.exists(dataset_dir):
#         raise FileNotFoundError(f"数据集路径不存在: {dataset_dir}")
    
#     dataset = XYDataset(dataset_dir, random_hflips=True)

#     # 训练集与测试集划分（保持不变）
#     test_percent = 0.1
#     num_test = int(test_percent * len(dataset))
#     train_dataset, test_dataset = torch.utils.data.random_split(
#         dataset, [len(dataset) - num_test, num_test])

#     train_loader = DataLoader(
#         train_dataset, batch_size=32, shuffle=True, num_workers=0)

#     test_loader = DataLoader(
#         test_dataset, batch_size=32, shuffle=False, num_workers=0)

#     # 初始化 MobileNet 模型
#     model = timm.create_model('mobilenetv2_100', pretrained=True)  # 使用 timm 加载 MobileNet
#     model.classifier = nn.Linear(model.classifier.in_features, 2)  # 替换分类头为回归层
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)

#     # 损失函数和优化器（保持不变）
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)

#     # 训练参数
#     num_epochs = 50
#     best_loss = float('inf')
#     model_save_path = './pth/best_mobilenet_model.pth'

#     print("start training...")
#     # 训练和测试循环（保持不变）
#     for epoch in range(num_epochs):
#         # 训练阶段
#         model.train()
#         train_loss = 0.0
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             train_loss += loss.item()
#             loss.backward()
#             optimizer.step()
#         train_loss /= len(train_loader)

#         # 测试阶段
#         model.eval()
#         test_loss = 0.0
#         with torch.no_grad():
#             for images, labels in test_loader:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
#                 test_loss += loss.item()
#         test_loss /= len(test_loader)

#         print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

#         # 保存最优模型
#         if test_loss < best_loss:
#             best_loss = test_loss
#             torch.save(model.state_dict(), model_save_path)
#             print(f"最佳模型已保存至 {model_save_path}")

# if __name__ == "__main__":
#     main()

import os

#os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

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
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime

# 数据集处理工具（保持不变）
def get_x(path, width):
    """根据图片文件名获取x坐标"""
    return (float(int(path.split("_")[1])) - width / 2) / (width / 2)

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
        x = float(get_x(os.path.basename(image_path), 640))
        y = float(get_y(os.path.basename(image_path), 480))
        
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
    # 创建TensorBoard日志目录
    log_dir = os.path.join('./runs', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard日志保存在: {log_dir}")
    
    dataset_dir = './annotated_images_1'  # 数据集路径
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"数据集路径不存在: {dataset_dir}")
    
    # 确保模型保存目录存在
    model_dir = './pth'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    dataset = XYDataset(dataset_dir, random_hflips=True)
    print(f"数据集样本总数: {len(dataset)}")

    # 训练集与测试集划分
    test_percent = 0.1
    num_test = int(test_percent * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [len(dataset) - num_test, num_test])
    
    print(f"训练集样本数: {len(train_dataset)}, 测试集样本数: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=0)

    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 初始化 MobileNet 模型
    model = timm.create_model('mobilenetv2_100', pretrained=True)  # 使用 timm 加载 MobileNet
    model.classifier = nn.Linear(model.classifier.in_features, 2)  # 替换分类头为回归层
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = model.to(device)

    # 将模型结构记录到TensorBoard
    dummy_input = torch.zeros(1, 3, 224, 224).to(device)
    writer.add_graph(model, dummy_input)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 训练参数
    num_epochs = 50
    best_loss = float('inf')
    model_save_path = os.path.join(model_dir, 'mobilenet_model_1.pth')

    print("开始训练...")
    # 训练和测试循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            # 更新进度条
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        # 测试阶段
        model.eval()
        test_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            test_progress_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Test]")
            for images, labels in test_progress_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                # 收集预测和标签用于可视化
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # 更新进度条
                test_progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        test_loss /= len(test_loader)
        
        # 应用学习率调度器
        scheduler.step(test_loss)

        # 记录损失到TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Test', test_loss, epoch)
        
        # 每5个epoch可视化预测结果
        if epoch % 5 == 0:
            # 将预测和标签转换为numpy数组
            preds_np = np.array(all_preds)
            labels_np = np.array(all_labels)
            
            # 创建预测vs真实值的散点图
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # X坐标预测vs真实
            ax1.scatter(labels_np[:, 0], preds_np[:, 0], alpha=0.5)
            ax1.plot([-1, 1], [-1, 1], 'r--')  # 理想线
            ax1.set_title('X Coordinate: Predicted vs True')
            ax1.set_xlabel('True')
            ax1.set_ylabel('Predicted')
            
            # Y坐标预测vs真实
            ax2.scatter(labels_np[:, 1], preds_np[:, 1], alpha=0.5)
            ax2.plot([-1, 1], [-1, 1], 'r--')  # 理想线
            ax2.set_title('Y Coordinate: Predicted vs True')
            ax2.set_xlabel('True')
            ax2.set_ylabel('Predicted')
            
            # 添加到TensorBoard
            writer.add_figure('Predictions_vs_Targets', fig, epoch)
            plt.close(fig)

        # 打印当前epoch的结果
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, LR: {current_lr:.6f}")

        # 保存最优模型
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
            }, model_save_path)
            print(f"✓ 最佳模型已保存至 {model_save_path}")

    # 训练完成后关闭TensorBoard writer
    writer.close()
    print("训练完成!")

if __name__ == "__main__":
    main()