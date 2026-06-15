# 模型目录

存放 TensorRT 序列化后的 Engine 文件。

## 文件说明

- `corner_slot.trt` - 集装箱锁孔检测模型 (YOLO 实例分割)
- `corner_slot_fp16.trt` - FP16 量化版本
- `corner_slot_int8.trt` - INT8 量化版本

## 模型生成

使用以下步骤生成 TensorRT Engine：

1. 准备 ONNX 模型
2. 使用 trtexec 转换：

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=corner_slot.onnx \
  --saveEngine=corner_slot.trt \
  --fp16 \
  --workspace=4096 \
  --verbose
```

或者使用 C++ API 编程生成。

## 输入输出规格

### 输入
- 尺寸: 640x640
- 格式: RGB, CHW
- 数据类型: FP32/FP16
- 归一化: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

### 输出
- 检测框: [x, y, w, h, conf, class_id]
- 实例分割 mask (可选)

## 性能

| 模型版本 | 精度 | FPS (Jetson Orin) | mAP |
|---------|------|------------------|-----|
| FP32 | 32bit | ~25 | 0.92 |
| FP16 | 16bit | ~50 | 0.91 |
| INT8 | 8bit | ~80 | 0.88 |
