[English](README.md)

# coreml-go

**coreml-go** 是一个 Go 语言的 CoreML 推理库，专为 macOS 上的模型推理而生。

> 本项目由 [gomlx/go-darwinml](https://github.com/gomlx/go-darwinml) 魔改而来。原项目侧重于模型构建与训练流程的封装，而 **coreml-go** 专注解决一个核心痛点：**直接加载已有的 .mlpackage / .mlmodelc 模型进行推理**，不含任何模型构建、训练或转换逻辑。

## 特性

- 加载编译后的 CoreML 模型（`.mlmodelc`），一步推理
- 支持编译 `.mlpackage` → `.mlmodelc`
- 支持图像输入（CVPixelBuffer）和多数组输入（Tensor）
- 支持 `float32`、`int32`、`int64`、`bool` 类型
- 零拷贝读取推理结果
- 无外部依赖（仅需 macOS + Go，基于 `cgo` 调用原生 CoreML 框架）

## 依赖

- macOS (CoreML 是 Apple 私有框架)
- Go 1.22+
- Xcode Command Line Tools（编译时需要，运行时不需要）

```bash
xcode-select --install
```

## 安装

```bash
go get github.com/your-username/go-coreml
```

## 极简入门

加载模型、构造虚拟输入、推理、获取结果：

```go
package main

import (
    "fmt"
    "go-coreml/bridge"
)

func main() {
    // Intel Mac 上避免推理崩溃；Apple Silicon 可不设置或用 ComputeAll
    bridge.SetComputeUnits(bridge.ComputeCPUOnly)

    // 1. 编译 .mlpackage → .mlmodelc（首次运行，后续可跳过）
    compiled, err := bridge.CompileModel("model.mlpackage", "")
    if err != nil {
        panic(err)
    }

    // 2. 加载模型
    model, err := bridge.LoadModel(compiled)
    if err != nil {
        panic(err)
    }
    defer model.Close()

    // 3. 构造输入张量（batch=1, 3, 640, 640）
    input, err := bridge.NewTensor[float32]([]int64{1, 3, 640, 640})
    if err != nil {
        panic(err)
    }
    defer input.Close()

    // 4. 构造输出张量
    output, err := bridge.NewTensor[float32]([]int64{1, 100, 84})
    if err != nil {
        panic(err)
    }
    defer output.Close()

    // 5. 推理
    err = model.Predict(
        []string{"image"}, []bridge.TensorHandle{input},
        []string{"output"}, []bridge.TensorHandle{output},
    )
    if err != nil {
        panic(err)
    }

    // 6. 读取结果（零拷贝）
    fmt.Println(output.Data())
}
```

### 图像输入推理

若模型输入层是图像类型（Image 而非 MultiArray），使用 `PredictWithImages`：

```go
imgInput, err := bridge.NewImageInput(640, 640, rgbaPix)
if err != nil {
    panic(err)
}
defer imgInput.Close()

output, err := bridge.NewTensor[float32]([]int64{1, 300, 6})
if err != nil {
    panic(err)
}
defer output.Close()

err = model.PredictWithImages(
    []string{"image"}, []*bridge.ImageInput{imgInput},
    []string{"var_1440"}, []bridge.TensorHandle{output},
)
```

## API

### 全局设置

| 函数 | 说明 |
|---|---|
| `bridge.SetComputeUnits(units)` | 设置计算单元。可选：`ComputeAll`（默认）、`ComputeCPUOnly`（兼容 Intel Mac）、`ComputeCPUAndGPU`、`ComputeCPUAndANE` |

### 模型编译与加载

| 函数 | 说明 |
|---|---|
| `bridge.CompileModel(packagePath, outputDir)` | 编译 `.mlpackage` 为 `.mlmodelc`，返回编译后路径 |
| `bridge.LoadModel(path)` | 加载 `.mlmodelc`，返回 `*Model` |

### 模型

| 方法 | 说明 |
|---|---|
| `model.Close()` | 释放模型资源 |
| `model.InputCount() int` | 返回模型输入数量 |
| `model.OutputCount() int` | 返回模型输出数量 |
| `model.InputName(index int) string` | 返回指定索引的输入名称 |
| `model.OutputName(index int) string` | 返回指定索引的输出名称 |
| `model.Predict(inputNames, inputs, outputNames, outputs)` | 执行推理，输入/输出均为 `TensorHandle` |
| `model.PredictWithImages(inputNames, images, outputNames, outputs)` | 图像输入推理，输入为 `*ImageInput`，输出为 `TensorHandle` |

### 张量

| 函数 / 方法 | 说明 |
|---|---|
| `bridge.NewTensor[T](shape)` | 创建类型化张量，`T` 支持 `float32`、`int32`、`int64`、`bool` |
| `tensor.Close()` | 释放张量资源 |
| `tensor.Rank() int` | 返回张量维度数 |
| `tensor.Dim(axis int) int64` | 返回指定维度的大小 |
| `tensor.Shape() []int64` | 返回张量形状 |
| `tensor.DType() DType` | 返回数据类型 |
| `tensor.SizeBytes() int64` | 返回数据总大小（字节） |
| `tensor.Data() []T` | 零拷贝获取底层数据切片 |
| `tensor.DataPtr() unsafe.Pointer` | 返回底层数据指针 |

### 图像输入

| 函数 / 方法 | 说明 |
|---|---|
| `bridge.NewImageInput(width, height, rgba)` | 从 RGBA 像素数据创建图像输入 |
| `imageInput.Close()` | 释放图像输入资源 |

## Intel Mac 兼容性

```go
bridge.SetComputeUnits(bridge.ComputeCPUOnly)
```

Intel 芯片的 Mac 上使用默认 `ComputeAll` 可能在部分模型推理时崩溃。设置 `ComputeCPUOnly` 可解决此问题。

## 示例

完整示例见 [examples/yolo26.go](examples/yolo26.go) —— 使用 Ultralytics 导出的 YOLO26 模型进行目标检测：

1. 用 `model.export()` 导出 CoreML 格式的 `.mlpackage`
2. 程序自动编译、加载模型
3. 读取图像，letterbox 缩放后推理
4. 解析输出**（300 个检测框）**

## 与原项目的差异

| | gomlx/go-darwinml | coreml-go |
|---|---|---|
| **定位** | 模型构建 + 训练 + 推理 | 推理专用 |
| **加载已有模型** | ❌ 不支持 | ✅ 支持 `.mlpackage` / `.mlmodelc` |
| **模型构建** | ✅ 包含 | ❌ 移除 |
| **训练流程** | ✅ 包含 | ❌ 移除 |
| **API 风格** | 面向构建链式调用 | 面向推理的简洁封装 |

