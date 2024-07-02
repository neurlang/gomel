package main

import (
	"fmt"
	"github.com/neurlang/gomel/mel"
	"os"
	"strconv"
)

func main() {
	// Check if the filename argument is provided
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run main.go <png_filename>")
		os.Exit(1)
	}

	// Get the filename from the command-line arguments
	var filename = os.Args[1]
	var freq = "44100"

	if len(os.Args) > 2 {
		freq = os.Args[2]
	}
	frequency, _ := strconv.Atoi(freq)

	// Create a new instance of Mel
	var m = mel.NewMel()

	// Set parameters
	m.NumMels = 512
	m.MelFmin = 0
	m.MelFmax = 8000
	m.YReverse = true
	m.Window = 256
	m.Resolut = 8192
	m.GriffinLimIterations = 2
	m.VolumeBoost = 0.0

	m.SampleRate = frequency

	// Generate the wave from a PNG file
	inputFile := filename
	outputFile := filename + ".wav"
	err := m.ToWavPng(inputFile, outputFile)
	if err != nil {
		fmt.Printf("Error generating wave from spectrogram: %v\n", err)
		os.Exit(1)
	}
}
