// Package bridge provides low-level cgo bindings to CoreML.
//
// This package wraps the Objective-C++ CoreML bridge and exposes
// Go-friendly APIs for model loading, tensor operations, and inference.
package bridge

/*
#cgo darwin CFLAGS: -fobjc-arc
#cgo darwin LDFLAGS: -framework Foundation -framework CoreML -framework CoreVideo
#include "bridge.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"reflect"
	"unsafe"
)

// DType represents a CoreML data type.
type DType int

const (
	DTypeFloat32 DType = C.COREML_DTYPE_FLOAT32
	DTypeFloat16 DType = C.COREML_DTYPE_FLOAT16
	DTypeInt32   DType = C.COREML_DTYPE_INT32
	DTypeInt64   DType = C.COREML_DTYPE_INT64
	DTypeBool    DType = C.COREML_DTYPE_BOOL
)

// ComputeUnits specifies which compute units to use.
type ComputeUnits int

const (
	ComputeAll       ComputeUnits = C.COREML_COMPUTE_ALL
	ComputeCPUOnly   ComputeUnits = C.COREML_COMPUTE_CPU_ONLY
	ComputeCPUAndGPU ComputeUnits = C.COREML_COMPUTE_CPU_AND_GPU
	ComputeCPUAndANE ComputeUnits = C.COREML_COMPUTE_CPU_AND_ANE
)

// SetComputeUnits sets the global compute units for model loading.
func SetComputeUnits(units ComputeUnits) {
	C.coreml_set_compute_units(C.CoreMLComputeUnits(units))
}

// Model represents a loaded CoreML model.
type Model struct {
	handle C.CoreMLModel
}

// CompileModel compiles an .mlpackage to .mlmodelc using CoreML.
// Returns the path to the compiled model.
// If outputDir is empty, the model is compiled to a temporary location.
func CompileModel(packagePath, outputDir string) (string, error) {
	cPackagePath := C.CString(packagePath)
	defer C.free(unsafe.Pointer(cPackagePath))
	var cOutputDir *C.char
	if outputDir != "" {
		cOutputDir = C.CString(outputDir)
		defer C.free(unsafe.Pointer(cOutputDir))
	}

	var err C.CoreMLError
	compiledPath := C.coreml_compile_model(cPackagePath, cOutputDir, &err)
	if compiledPath == nil {
		msg := "unknown error"
		if err.message != nil {
			msg = C.GoString(err.message)
			C.free(unsafe.Pointer(err.message))
		}
		return "", fmt.Errorf("failed to compile model: %s", msg)
	}

	result := C.GoString(compiledPath)
	C.free(unsafe.Pointer(compiledPath))
	return result, nil
}

// LoadModel loads a CoreML model from a .mlmodelc directory.
func LoadModel(path string) (*Model, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var err C.CoreMLError
	handle := C.coreml_load_model(cPath, &err)
	if handle == nil {
		msg := "unknown error"
		if err.message != nil {
			msg = C.GoString(err.message)
			C.free(unsafe.Pointer(err.message))
		}
		return nil, fmt.Errorf("failed to load model: %s", msg)
	}

	return &Model{handle: handle}, nil
}

// Close releases the model resources.
func (m *Model) Close() {
	if m.handle != nil {
		C.coreml_free_model(m.handle)
		m.handle = nil
	}
}

// InputCount returns the number of inputs.
func (m *Model) InputCount() int {
	return int(C.coreml_model_input_count(m.handle))
}

// OutputCount returns the number of outputs.
func (m *Model) OutputCount() int {
	return int(C.coreml_model_output_count(m.handle))
}

// InputName returns the name of the input at the given index.
func (m *Model) InputName(index int) string {
	cName := C.coreml_model_input_name(m.handle, C.int(index))
	if cName == nil {
		return ""
	}
	name := C.GoString(cName)
	C.free(unsafe.Pointer(cName))
	return name
}

// OutputName returns the name of the output at the given index.
func (m *Model) OutputName(index int) string {
	cName := C.coreml_model_output_name(m.handle, C.int(index))
	if cName == nil {
		return ""
	}
	name := C.GoString(cName)
	C.free(unsafe.Pointer(cName))
	return name
}

// TensorHandle provides access to a tensor's C handle and metadata.
// Both *Tensor[float32] and *Tensor[int32] implement this.
type TensorHandle interface {
	ptr() C.CoreMLTensor
	Shape() []int64
	DType() DType
}

// Tensor represents a typed multi-dimensional array for CoreML.
// T can be any Go type that has a corresponding CoreML DType
// (float32, int32, int64, bool).
type Tensor[T CoreMLType] struct {
	handle  C.CoreMLTensor
	data    []T
	dataPtr unsafe.Pointer
}

func (t *Tensor[T]) ptr() C.CoreMLTensor { return t.handle }

// newTensorRaw creates a tensor with the given shape and data type (internal).
func newTensorRaw[T CoreMLType](shape []int64, dtype DType) (*Tensor[T], error) {
	var err C.CoreMLError
	var shapePtr *C.int64_t
	if len(shape) > 0 {
		shapePtr = (*C.int64_t)(unsafe.Pointer(&shape[0]))
	}
	handle := C.coreml_tensor_create(
		shapePtr,
		C.int(len(shape)),
		C.int(dtype),
		&err,
	)
	if handle == nil {
		msg := "unknown error"
		if err.message != nil {
			msg = C.GoString(err.message)
			C.free(unsafe.Pointer(err.message))
		}
		return nil, fmt.Errorf("failed to create tensor: %s", msg)
	}

	dataPtr := C.coreml_tensor_data(handle)
	size := int64(1)
	for _, dim := range shape {
		size *= dim
	}
	data := unsafe.Slice((*T)(dataPtr), int(size))

	return &Tensor[T]{handle: handle, data: data, dataPtr: dataPtr}, nil
}

// NewTensor creates a new tensor with the given shape. The element type is
// determined by the type parameter T (must be float32 or int32).
//
// Usage:
//
//	t, err := bridge.NewTensor[float32]([]int64{1, 3, 640, 640})
func NewTensor[T CoreMLType](shape []int64) (*Tensor[T], error) {
	return newTensorRaw[T](shape, dtypeFor[T]())
}

// Close releases the tensor resources.
func (t *Tensor[T]) Close() {
	if t.handle != nil {
		C.coreml_tensor_free(t.handle)
		t.handle = nil
	}
}

// Rank returns the number of dimensions.
func (t *Tensor[T]) Rank() int {
	return int(C.coreml_tensor_rank(t.handle))
}

// Dim returns the size of the given dimension.
func (t *Tensor[T]) Dim(axis int) int64 {
	return int64(C.coreml_tensor_dim(t.handle, C.int(axis)))
}

// Shape returns the shape as a slice.
func (t *Tensor[T]) Shape() []int64 {
	rank := t.Rank()
	shape := make([]int64, rank)
	for i := range rank {
		shape[i] = t.Dim(i)
	}
	return shape
}

// DType returns the data type.
func (t *Tensor[T]) DType() DType {
	return DType(C.coreml_tensor_dtype(t.handle))
}

// DataPtr returns an unsafe pointer to the underlying data.
func (t *Tensor[T]) DataPtr() unsafe.Pointer {
	return t.dataPtr
}

// SizeBytes returns the total size in bytes.
func (t *Tensor[T]) SizeBytes() int64 {
	return int64(C.coreml_tensor_size_bytes(t.handle))
}

// Predict runs inference with the given tensor inputs and outputs.
func (m *Model) Predict(inputNames []string, inputs []TensorHandle, outputNames []string, outputs []TensorHandle) error {
	if len(inputNames) != len(inputs) {
		return fmt.Errorf("input names count (%d) != inputs count (%d)", len(inputNames), len(inputs))
	}
	if len(outputNames) != len(outputs) {
		return fmt.Errorf("output names count (%d) != outputs count (%d)", len(outputNames), len(outputs))
	}

	cInputNames := make([]*C.char, len(inputNames))
	for i, name := range inputNames {
		cInputNames[i] = C.CString(name)
	}
	defer func() {
		for _, name := range cInputNames {
			C.free(unsafe.Pointer(name))
		}
	}()

	cOutputNames := make([]*C.char, len(outputNames))
	for i, name := range outputNames {
		cOutputNames[i] = C.CString(name)
	}
	defer func() {
		for _, name := range cOutputNames {
			C.free(unsafe.Pointer(name))
		}
	}()

	cInputs := make([]C.CoreMLTensor, len(inputs))
	for i, t := range inputs {
		cInputs[i] = t.ptr()
	}

	cOutputs := make([]C.CoreMLTensor, len(outputs))
	for i, t := range outputs {
		cOutputs[i] = t.ptr()
	}

	var cInputNamesPtr **C.char
	var cInputsPtr *C.CoreMLTensor
	if len(inputs) > 0 {
		cInputNamesPtr = (**C.char)(unsafe.Pointer(&cInputNames[0]))
		cInputsPtr = (*C.CoreMLTensor)(unsafe.Pointer(&cInputs[0]))
	}

	var cOutputNamesPtr **C.char
	var cOutputsPtr *C.CoreMLTensor
	if len(outputs) > 0 {
		cOutputNamesPtr = (**C.char)(unsafe.Pointer(&cOutputNames[0]))
		cOutputsPtr = (*C.CoreMLTensor)(unsafe.Pointer(&cOutputs[0]))
	}

	var err C.CoreMLError
	ok := C.coreml_model_predict(
		m.handle,
		cInputNamesPtr,
		cInputsPtr,
		C.int(len(inputs)),
		cOutputNamesPtr,
		cOutputsPtr,
		C.int(len(outputs)),
		&err,
	)

	if !ok {
		msg := "unknown error"
		if err.message != nil {
			msg = C.GoString(err.message)
			C.free(unsafe.Pointer(err.message))
		}
		return fmt.Errorf("prediction failed: %s", msg)
	}

	return nil
}

// ImageInput wraps a CVPixelBuffer for models that expect image-type inputs.
type ImageInput struct {
	handle C.CoreMLImageInput
}

// NewImageInput creates an ImageInput from uint8 RGBA pixel data.
// The rgba slice must have length width*height*4.
// The data is copied internally, so the caller may reuse rgba after this returns.
func NewImageInput(width, height int, rgba []uint8) (*ImageInput, error) {
	if len(rgba) < width*height*4 {
		return nil, fmt.Errorf("rgba data too short: need %d, got %d", width*height*4, len(rgba))
	}
	var err C.CoreMLError
	handle := C.coreml_image_input_create(
		C.int(width), C.int(height),
		unsafe.Pointer(&rgba[0]),
		&err,
	)
	if handle == nil {
		msg := "unknown error"
		if err.message != nil {
			msg = C.GoString(err.message)
			C.free(unsafe.Pointer(err.message))
		}
		return nil, fmt.Errorf("failed to create image input: %s", msg)
	}
	return &ImageInput{handle: handle}, nil
}

// Close releases the image input resources.
func (ii *ImageInput) Close() {
	if ii.handle != nil {
		C.coreml_image_input_free(ii.handle)
		ii.handle = nil
	}
}

// PredictWithImages runs inference with image-type inputs (CVPixelBuffer).
// Use this when the model expects image features rather than multi-array inputs.
func (m *Model) PredictWithImages(inputNames []string, images []*ImageInput, outputNames []string, outputs []TensorHandle) error {
	if len(inputNames) != len(images) {
		return fmt.Errorf("input names count (%d) != images count (%d)", len(inputNames), len(images))
	}
	if len(outputNames) != len(outputs) {
		return fmt.Errorf("output names count (%d) != outputs count (%d)", len(outputNames), len(outputs))
	}

	cInputNames := make([]*C.char, len(inputNames))
	for i, name := range inputNames {
		cInputNames[i] = C.CString(name)
	}
	defer func() {
		for _, name := range cInputNames {
			C.free(unsafe.Pointer(name))
		}
	}()

	cOutputNames := make([]*C.char, len(outputNames))
	for i, name := range outputNames {
		cOutputNames[i] = C.CString(name)
	}
	defer func() {
		for _, name := range cOutputNames {
			C.free(unsafe.Pointer(name))
		}
	}()

	cImages := make([]C.CoreMLImageInput, len(images))
	for i, img := range images {
		cImages[i] = img.handle
	}

	cOutputs := make([]C.CoreMLTensor, len(outputs))
	for i, t := range outputs {
		cOutputs[i] = t.ptr()
	}

	var cInputNamesPtr **C.char
	var cImagesPtr *C.CoreMLImageInput
	if len(images) > 0 {
		cInputNamesPtr = (**C.char)(unsafe.Pointer(&cInputNames[0]))
		cImagesPtr = (*C.CoreMLImageInput)(unsafe.Pointer(&cImages[0]))
	}

	var cOutputNamesPtr **C.char
	var cOutputsPtr *C.CoreMLTensor
	if len(outputs) > 0 {
		cOutputNamesPtr = (**C.char)(unsafe.Pointer(&cOutputNames[0]))
		cOutputsPtr = (*C.CoreMLTensor)(unsafe.Pointer(&cOutputs[0]))
	}

	var err C.CoreMLError
	ok := C.coreml_model_predict_with_images(
		m.handle,
		cInputNamesPtr,
		cImagesPtr,
		C.int(len(images)),
		cOutputNamesPtr,
		cOutputsPtr,
		C.int(len(outputs)),
		&err,
	)

	if !ok {
		msg := "unknown error"
		if err.message != nil {
			msg = C.GoString(err.message)
			C.free(unsafe.Pointer(err.message))
		}
		return fmt.Errorf("prediction with images failed: %s", msg)
	}

	return nil
}

// CoreMLType represents Go types that have a corresponding CoreML data type.
type CoreMLType interface {
	~float32 | ~int32 | ~int64 | ~bool
}

// dtypeFor returns the bridge DType for a generic type parameter.
func dtypeFor[T CoreMLType]() DType {
	switch reflect.TypeFor[T]().Kind() {
	case reflect.Float32:
		return DTypeFloat32
	case reflect.Int32:
		return DTypeInt32
	case reflect.Int64:
		return DTypeInt64
	case reflect.Bool:
		return DTypeBool
	default:
		panic(fmt.Sprintf("unsupported CoreML type: %s", reflect.TypeFor[T]()))
	}
}

// Data returns a zero-copy view of the tensor's underlying data.
// The slice is valid only while the tensor is alive (not closed).
func (t *Tensor[T]) Data() []T {
	return t.data
}
