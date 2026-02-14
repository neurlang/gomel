package main

import (
	"fmt"
	"github.com/neurlang/gomel/phase"
	"os"
	"strings"
)

func main() {
	// Check if the filename argument is provided
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run main.go <wav_base_filename>")
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

	if strings.HasSuffix(filename, ".flac") {
		// Generate the mel spectrogram and save it as a PNG file
		inputFile := filename
		outputFile := filename + ".png"
		err := m.ToPhaseFlac(inputFile, outputFile)
		if err != nil {
			fmt.Printf("Error generating mel spectrogram: %v\n", err)
			os.Exit(1)
		}
	} else if strings.HasSuffix(filename, ".wav") {
		// Generate the mel spectrogram and save it as a PNG file
		inputFile := filename
		outputFile := filename + ".png"
		err := m.ToPhaseWav(inputFile, outputFile)
		if err != nil {
			fmt.Printf("Error generating mel spectrogram: %v\n", err)
			os.Exit(1)
		}
	} else {
		// Generate the mel spectrogram and save it as a PNG file
		inputFile := filename + ".wav"
		outputFile := filename + ".png"
		err := m.ToPhaseWav(inputFile, outputFile)
		if err != nil {
			fmt.Printf("Error generating mel spectrogram: %v\n", err)
			os.Exit(1)
		}
	}
}
