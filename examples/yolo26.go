package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"

	"golang.org/x/image/draw"

	"github.com/jzksnsjswkw/coreml-go/bridge"
)

type BoundingBox struct {
	ClassID        int
	Confidence     float32
	X1, Y1, X2, Y2 float32
}

func main() {
	// Intel Mac stability; Apple Silicon can use bridge.ComputeAll for speed
	bridge.SetComputeUnits(bridge.ComputeCPUOnly)

	// ── Compile .mlpackage ──────────────────────────────────
	const compiledName = "best.mlmodelc"
	if _, err := os.Stat(compiledName); os.IsNotExist(err) {
		compiledPath, err := bridge.CompileModel("best.mlpackage", "")
		if err != nil {
			panic(err)
		}
		if err := os.Rename(compiledPath, compiledName); err != nil {
			panic(err)
		}
	} else if err != nil {
		panic(err)
	}

	// ── Load model ──────────────────────────────────────────
	model, err := bridge.LoadModel(compiledName)
	if err != nil {
		panic(err)
	}
	defer model.Close()

	// ── Load image ──────────────────────────────────────────
	f, err := os.Open("img.png")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		panic(err)
	}

	// ── Letterbox resize ───────────
	const inputW, inputH = 640, 640
	bounds := img.Bounds()
	srcW, srcH := bounds.Dx(), bounds.Dy()

	ratio := math.Min(float64(inputW)/float64(srcW), float64(inputH)/float64(srcH))
	newW := int(math.Round(float64(srcW) * ratio))
	newH := int(math.Round(float64(srcH) * ratio))
	left := (inputW - newW) / 2
	top := (inputH - newH) / 2

	dst := image.NewRGBA(image.Rect(0, 0, inputW, inputH))
	draw.Draw(dst, dst.Bounds(), &image.Uniform{color.RGBA{114, 114, 114, 255}}, image.Point{}, draw.Src)
	draw.BiLinear.Scale(dst, image.Rect(left, top, left+newW, top+newH), img, bounds, draw.Over, nil)

	// ── CoreML image input ──────────────────────────────────
	imgInput, err := bridge.NewImageInput(inputW, inputH, dst.Pix)
	if err != nil {
		panic(err)
	}
	defer imgInput.Close()

	// ── Output tensor (300 detections × [x1,y1,x2,y2,conf,class_id]) ─
	output, err := bridge.NewTensor[float32]([]int64{1, 300, 6})
	if err != nil {
		panic(err)
	}
	defer output.Close()

	// ── Inference ───────────────────────────────────────────
	err = model.PredictWithImages(
		[]string{"image"}, []*bridge.ImageInput{imgInput},
		[]string{"var_1440"}, []bridge.TensorHandle{output},
	)
	if err != nil {
		panic(err)
	}

	// ── Parse detections ────────────────────────────────────
	data := output.Data()
	confThresh := float32(0.25)
	boxes := make([]BoundingBox, 0, 300)

	for i := range 300 {
		off := i * 6
		conf := data[off+4]
		if conf < confThresh {
			continue
		}
		boxes = append(boxes, BoundingBox{
			ClassID:    int(data[off+5]),
			Confidence: conf,
			X1:         (data[off+0] - float32(left)) / float32(ratio),
			Y1:         (data[off+1] - float32(top)) / float32(ratio),
			X2:         (data[off+2] - float32(left)) / float32(ratio),
			Y2:         (data[off+3] - float32(top)) / float32(ratio),
		})
	}

	fmt.Printf("Detected %d objects\n", len(boxes))
	for _, b := range boxes {
		fmt.Printf("  class=%d conf=%f  [%f, %f, %f, %f]\n",
			b.ClassID, b.Confidence, b.X1, b.Y1, b.X2, b.Y2)
	}
}
