// Package bridge provides low-level cgo bindings to CoreML.
package bridge

import "testing"

func TestTensorCreate(t *testing.T) {
	shape := []int64{2, 3}
	tensor, err := NewTensor[float32](shape)
	if err != nil {
		t.Fatalf("NewTensor failed: %v", err)
	}
	defer tensor.Close()

	if tensor.Rank() != 2 {
		t.Errorf("expected rank 2, got %d", tensor.Rank())
	}
	if tensor.Dim(0) != 2 {
		t.Errorf("expected dim 0 = 2, got %d", tensor.Dim(0))
	}
	if tensor.Dim(1) != 3 {
		t.Errorf("expected dim 1 = 3, got %d", tensor.Dim(1))
	}
	if tensor.DType() != DTypeFloat32 {
		t.Errorf("expected dtype Float32, got %d", tensor.DType())
	}
	if tensor.SizeBytes() != 2*3*4 {
		t.Errorf("expected size 24 bytes, got %d", tensor.SizeBytes())
	}
}

func TestTensorData(t *testing.T) {
	tensor, err := NewTensor[float32]([]int64{2, 2})
	if err != nil {
		t.Fatalf("NewTensor failed: %v", err)
	}
	defer tensor.Close()

	data := []float32{1, 2, 3, 4}
	src := (*[4]float32)(tensor.DataPtr())[:4:4]
	copy(src, data)

	got := tensor.Data()
	if len(got) != 4 {
		t.Fatalf("expected 4 elements, got %d", len(got))
	}
	for i, v := range data {
		if got[i] != v {
			t.Errorf("data[%d] = %f, expected %f", i, got[i], v)
		}
	}
}

func TestTensorShape(t *testing.T) {
	shape := []int64{3, 4, 5}
	tensor, err := NewTensor[float32](shape)
	if err != nil {
		t.Fatalf("NewTensor failed: %v", err)
	}
	defer tensor.Close()

	got := tensor.Shape()
	if len(got) != len(shape) {
		t.Fatalf("expected shape length %d, got %d", len(shape), len(got))
	}
	for i, v := range shape {
		if got[i] != v {
			t.Errorf("shape[%d] = %d, expected %d", i, got[i], v)
		}
	}
}

func TestComputeUnits(t *testing.T) {
	SetComputeUnits(ComputeAll)
	SetComputeUnits(ComputeCPUOnly)
	SetComputeUnits(ComputeCPUAndGPU)
	SetComputeUnits(ComputeCPUAndANE)
	SetComputeUnits(ComputeAll)
}
