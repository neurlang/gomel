package main

import (
	"fmt"
	"github.com/neurlang/gomel/mel"
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

	// Create a new instance of Mel
	var m = mel.NewMel()

	// Set parameters
	m.MelFmin = 0
	m.MelFmax = 16000
	m.YReverse = true
	m.Window = 1280
	m.NumMels = 192
	m.Resolut = 4096
	m.GriffinLimIterations = 2
	m.VolumeBoost = 0.0

	if strings.HasSuffix(filename, ".flac") {
		// Generate the mel spectrogram and save it as a PNG file
		inputFile := filename
		outputFile := filename + ".png"
		err := m.ToMelFlac(inputFile, outputFile)
		if err != nil {
			fmt.Printf("Error generating mel spectrogram: %v\n", err)
			os.Exit(1)
		}
	} else if strings.HasSuffix(filename, ".wav") {
		// Generate the mel spectrogram and save it as a PNG file
		inputFile := filename
		outputFile := filename + ".png"
		err := m.ToMelWav(inputFile, outputFile)
		if err != nil {
			fmt.Printf("Error generating mel spectrogram: %v\n", err)
			os.Exit(1)
		}
	} else {
		// Generate the mel spectrogram and save it as a PNG file
		inputFile := filename + ".wav"
		outputFile := filename + ".png"
		err := m.ToMelWav(inputFile, outputFile)
		if err != nil {
			fmt.Printf("Error generating mel spectrogram: %v\n", err)
			os.Exit(1)
		}
	}
}
