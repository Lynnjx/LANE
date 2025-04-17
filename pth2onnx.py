import torchvision
import torch
import timm

def main(args=None):
  model = timm.create_model('vit_base_patch16_224', pretrained=False)
  model.head = torch.nn.Linear(model.head.in_features, 2)
  model.load_state_dict(torch.load('./pth/best_vit_model.pth'))
  device = torch.device('cpu')
  model = model.to(device)
  model.eval()
  x = torch.randn(1, 3, 224, 224, requires_grad=True)
  torch_out = model(x)
  torch.onnx.export(model,
                    x,
                    "./onnx/best_vit_model_xy.onnx",
                    export_params=True,
                    opset_version=14,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'])

if __name__ == '__main__':
  main()