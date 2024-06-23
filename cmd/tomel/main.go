package main

import (
	"fmt"
	"os"
	"localhost/gomel/mel" // make sure this is the correct import path for your `libmel` package
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
	var m = libmel.NewMel()

	// Set parameters
	m.NumMels = 80
	m.MelFmin = 0
	m.MelFmax = 8000
	m.YReverse = true
	m.Window = 1024
	m.Resolut = 8192

	// Generate the mel spectrogram and save it as a PNG file
	inputFile := filename + ".wav"
	outputFile := filename + ".png"
	err := m.ToMel(inputFile, outputFile)
	if err != nil {
		fmt.Printf("Error generating mel spectrogram: %v\n", err)
		os.Exit(1)
	}
}
