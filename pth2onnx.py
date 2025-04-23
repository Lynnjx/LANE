import os
import torchvision
import torch
import timm

def main(args=None):
  model = timm.create_model('mobilenetv2_100', pretrained=False)
  model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
  # model.load_state_dict(torch.load('./pth/best_mobilenet_model.pth'))
  checkpoint = torch.load('./pth/mobilenet_model_1.pth')
  if 'model_state_dict' in checkpoint:
    # 仅加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"成功加载模型 (epoch: {checkpoint['epoch']})")
  else:
    # 如果是直接的模型状态字典
    model.load_state_dict(checkpoint)
    print("成功加载模型")
  device = torch.device('cpu')
  model = model.to(device)
  model.eval()
  x = torch.randn(1, 3, 224, 224, requires_grad=True)
  with torch.no_grad():
    torch_out = model(x)
    print(f"模型输出形状: {torch_out.shape}")
  torch.onnx.export(model,
                    x,
                    "./onnx/mobilenet_model_1_xy.onnx",
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'])
  print(f"模型已成功导出至: ./onnx/best_mobilenet_model_xy.onnx")

if __name__ == '__main__':
  main()
