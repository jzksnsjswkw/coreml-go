[中文](README_CN.md)

# coreml-go

**coreml-go** is a Go inference library for CoreML, built for running model inference on macOS.

> This project is derived from [gomlx/go-darwinml](https://github.com/gomlx/go-darwinml). The original project focuses on model construction and training pipeline wrappers, while **coreml-go** solves one core problem: **directly loading existing `.mlpackage` / `.mlmodelc` models for inference** — no model building, training, or conversion logic included.

## Features

- Load compiled CoreML models (`.mlmodelc`) and run inference in one step
- Compile `.mlpackage` → `.mlmodelc`
- Support image inputs (CVPixelBuffer) and multi-array inputs (Tensor)
- Support `float32`, `int32`, `int64`, `bool` types
- Zero-copy reads for inference results
- No external dependencies (only macOS + Go, powered by `cgo` calling native CoreML framework)

## Requirements

- macOS (CoreML is an Apple private framework)
- Go 1.22+
- Xcode Command Line Tools (required at build time only)

```bash
xcode-select --install
```

## Installation

```bash
go get github.com/your-username/go-coreml
```

## Quick Start

Load a model, create dummy inputs, run inference, and read results:

```go
package main

import (
    "fmt"
    "go-coreml/bridge"
)

func main() {
    // Prevents crashes on Intel Macs; Apple Silicon can use ComputeAll or omit
    bridge.SetComputeUnits(bridge.ComputeCPUOnly)

    // 1. Compile .mlpackage → .mlmodelc (first run only, can skip afterward)
    compiled, err := bridge.CompileModel("model.mlpackage", "")
    if err != nil {
        panic(err)
    }

    // 2. Load model
    model, err := bridge.LoadModel(compiled)
    if err != nil {
        panic(err)
    }
    defer model.Close()

    // 3. Create input tensor (batch=1, 3, 640, 640)
    input, err := bridge.NewTensor[float32]([]int64{1, 3, 640, 640})
    if err != nil {
        panic(err)
    }
    defer input.Close()

    // 4. Create output tensor
    output, err := bridge.NewTensor[float32]([]int64{1, 100, 84})
    if err != nil {
        panic(err)
    }
    defer output.Close()

    // 5. Run inference
    err = model.Predict(
        []string{"image"}, []bridge.TensorHandle{input},
        []string{"output"}, []bridge.TensorHandle{output},
    )
    if err != nil {
        panic(err)
    }

    // 6. Read results (zero-copy)
    fmt.Println(output.Data())
}
```

### Image Input Inference

If the model's input layer is an Image type (rather than MultiArray), use `PredictWithImages`:

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

### Global Settings

| Function | Description |
|---|---|
| `bridge.SetComputeUnits(units)` | Set compute units. Options: `ComputeAll` (default), `ComputeCPUOnly` (compatible with Intel Mac), `ComputeCPUAndGPU`, `ComputeCPUAndANE` |

### Model Compilation & Loading

| Function | Description |
|---|---|
| `bridge.CompileModel(packagePath, outputDir)` | Compile `.mlpackage` to `.mlmodelc`, returns compiled path |
| `bridge.LoadModel(path)` | Load a `.mlmodelc`, returns `*Model` |

### Model

| Method | Description |
|---|---|
| `model.Close()` | Release model resources |
| `model.InputCount() int` | Return the number of model inputs |
| `model.OutputCount() int` | Return the number of model outputs |
| `model.InputName(index int) string` | Return the input name at the given index |
| `model.OutputName(index int) string` | Return the output name at the given index |
| `model.Predict(inputNames, inputs, outputNames, outputs) error` | Run inference with tensor inputs/outputs |
| `model.PredictWithImages(inputNames, images, outputNames, outputs) error` | Image input inference; inputs are `*ImageInput`, outputs are `TensorHandle` |

### Tensors

| Function / Method | Description |
|---|---|
| `bridge.NewTensor[T](shape)` | Create a typed tensor. `T` supports `float32`, `int32`, `int64`, `bool` |
| `tensor.Close()` | Release tensor resources |
| `tensor.Rank() int` | Return the number of dimensions |
| `tensor.Dim(axis int) int64` | Return the size of the given dimension |
| `tensor.Shape() []int64` | Return the tensor shape |
| `tensor.DType() DType` | Return the data type |
| `tensor.SizeBytes() int64` | Return the total data size in bytes |
| `tensor.Data() []T` | Return a zero-copy view of the underlying data slice |
| `tensor.DataPtr() unsafe.Pointer` | Return an unsafe pointer to the underlying data |

### Image Input

| Function / Method | Description |
|---|---|
| `bridge.NewImageInput(width, height, rgba)` | Create an image input from RGBA pixel data |
| `imageInput.Close()` | Release image input resources |

## Intel Mac Compatibility

```go
bridge.SetComputeUnits(bridge.ComputeCPUOnly)
```

On Intel Macs, the default `ComputeAll` may cause crashes with certain models. Setting `ComputeCPUOnly` resolves this issue.

## Example

A complete example is available at [examples/yolo26.go](examples/yolo26.go) — object detection using a YOLO26 model exported from Ultralytics:

1. Export the model to CoreML `.mlpackage` with `model.export()`
2. The program automatically compiles and loads the model
3. Reads an image, applies letterbox resizing, and runs inference
4. Parses the output (300 detections)

## Differences from the Original Project

| | gomlx/go-darwinml | coreml-go |
|---|---|---|
| **Focus** | Model building + training + inference | Inference only |
| **Load existing models** | ❌ Not supported | ✅ Supports `.mlpackage` / `.mlmodelc` |
| **Model building** | ✅ Included | ❌ Removed |
| **Training pipeline** | ✅ Included | ❌ Removed |
| **API style** | Builder chain API | Minimal inference-focused API |

