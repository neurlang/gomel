package main

import (
	"fmt"
	"github.com/neurlang/gomel/mel"
	"os"
)

func main() {
	// Check if the filename argument is provided
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run main.go <png_filename>")
		os.Exit(1)
	}

	// Get the filename from the command-line arguments
	var filename = os.Args[1]

	// Create a new instance of Mel
	var m = mel.NewMel()

	// Set parameters
	m.NumMels = 160
	m.MelFmin = 0
	m.MelFmax = 8000
	m.YReverse = true
	m.Window = 1024
	m.Resolut = 8192
	m.GriffinLimIterations = 5
	m.Spread = -13
	// Generate the wave from a PNG file
	inputFile := filename
	outputFile := filename + ".wav"
	err := m.ToWavPng(inputFile, outputFile)
	if err != nil {
		fmt.Printf("Error generating wave from spectrogram: %v\n", err)
		os.Exit(1)
	}
}
