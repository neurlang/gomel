package main

import (
	"fmt"
	"github.com/neurlang/gomel/phase"
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

	// Create a new instance of Phase
	var m = phase.NewPhase()

	// Set parameters to match Python defaults
	m.YReverse = true
	m.NumFreqs = 768
	m.Window = 1280
	m.Resolut = 4096
	m.VolumeBoost = 4

	// Generate the wave from a PNG file
	inputFile := filename
	outputFile := filename + ".wav"
	err := m.ToWavPng(inputFile, outputFile)
	if err != nil {
		fmt.Printf("Error generating wave from spectrogram: %v\n", err)
		os.Exit(1)
	}
}
