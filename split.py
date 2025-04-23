import os
import random
import shutil
import glob

def split_test_images(src_dir='./annotated_images_1', 
                       test_dir='./annotated_test_1', 
                       num_images=30):
    """
    从标注图像文件夹中随机提取一定数量的图片作为测试集，并从源文件夹中删除
    
    参数:
        src_dir: 源数据目录
        test_dir: 测试数据输出目录
        num_images: 提取的测试图片数量
    """
    # 确保源目录存在
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"源目录不存在: {src_dir}")
    
    # 创建测试目录
    os.makedirs(test_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_paths = glob.glob(os.path.join(src_dir, "*.jpg"))
    
    if not image_paths:
        print(f"在 {src_dir} 中没有找到图像文件")
        return []
    
    total_images = len(image_paths)
    print(f"源文件夹中共有 {total_images} 张图片")
    
    # 确保请求的图片数量不超过可用图片总数
    num_images = min(num_images, total_images)
    
    # 随机选择指定数量的图片
    selected_images = random.sample(image_paths, num_images)
    
    moved_files = []
    
    # 移动选定的文件到测试文件夹
    print(f"正在将 {num_images} 个文件移动到测试集目录...")
    for img_path in selected_images:
        filename = os.path.basename(img_path)
        dst_path = os.path.join(test_dir, filename)
        
        # 使用shutil.move移动文件（而非复制）
        shutil.move(img_path, dst_path)
        moved_files.append(filename)
    
    # 计算移动后的文件数量
    remaining_files = len(glob.glob(os.path.join(src_dir, "*.jpg")))
    
    print(f"完成!")
    print(f"- {num_images} 张图片已移动到测试集: {os.path.abspath(test_dir)}")
    print(f"- 训练集中剩余 {remaining_files} 张图片")
    
    return moved_files

if __name__ == "__main__":
    # 设置随机种子以确保结果可重现（可以注释掉此行以获取随机结果）
    random.seed(42)
    
    # 提取30张测试图片
    moved_files = split_test_images(
        src_dir='./annotated_images_1',
        test_dir='./annotated_test_1',
        num_images=30
    )
    
    # 打印移动的文件名（前5个）
    if moved_files:
        print("\n已移动的部分测试文件:")
        for i, file in enumerate(moved_files[:5]):
            print(f"  {i+1}. {file}")
        if len(moved_files) > 5:
            print(f"  ... 以及 {len(moved_files)-5} 个其他文件")