package mel

import "github.com/r9y9/gossp/stft"
import "github.com/mjibson/go-dsp/fft"
import "errors"
import "math/cmplx"

//import "math/rand"

// Mel represents the configuration for generating mel spectrograms.
type Mel struct {
	NumMels  int
	MelFmin  float64
	MelFmax  float64
	TuneMul  float64
	TuneAdd  float64
	Window   int
	Resolut  int
	YReverse bool

	GriffinLimIterations int

	// VolumeBoost when loading spectrogram from image, can be a value like 1.666
	VolumeBoost float64

	// sample rate for output wav
	SampleRate int
}

// NewMel creates a new Mel instance with default values.
func NewMel() *Mel {
	return &Mel{
		NumMels:              160,
		MelFmin:              0,
		MelFmax:              8000,
		TuneMul:              1,
		TuneAdd:              0,
		Window:               256,
		Resolut:              2048,
		GriffinLimIterations: 2,
	}
}

var ErrFileNotLoaded = errors.New("wavNotLoaded")

// ToMel generates a mel spectrogram from a wave buffer and returns the mel buffer.
func (m *Mel) ToMel(buf []float64) ([][2]float64, error) {

	buf = pad(buf, m.Window)

	stft := stft.New(m.Window, m.Resolut)

	spectrum := stft.STFT(buf)
	var ospectrum [][2]float64
	for i := range spectrum {
		for j := 0; j < m.Resolut/2; j++ {

			var v0 = spectrum[i][j]
			var v1 = spectrum[i][m.Resolut-j-1]

			var realn0 = cmplx.Abs(v0)
			var realn1 = cmplx.Abs(v1)

			ospectrum = append(ospectrum, [2]float64{realn0, realn1})

		}
	}

	ospectrum = domel(m.Resolut/2, m.NumMels, ospectrum, m.MelFmin, m.MelFmax)

	spectral_normalize(ospectrum)

	return ospectrum, nil

}

func ISTFT(s *stft.STFT, spectrogram [][]complex128, numIterations int) []float64 {
	frameShift := s.FrameShift
	frameLen := len(spectrogram[0])
	numFrames := len(spectrogram)
	reconstructedSignal := make([]float64, frameLen+(numFrames-1)*frameShift)
	windowSum := make([]float64, frameLen+(numFrames-1)*frameShift)

	// Griffin-Lim iterations
	for iter := 0; iter < numIterations; iter++ {
		// Calculate the STFT of the reconstructed signal
		for i := 0; i < numFrames; i++ {
			frame := make([]float64, frameLen)
			for j := 0; j < frameLen; j++ {
				if i*frameShift+j < len(reconstructedSignal) {
					frame[j] = reconstructedSignal[i*frameShift+j] * s.Window[j]
				}
			}
			stftFrame := fft.FFTReal(frame)

			// Update the phase of the spectrogram with the phase of the current STFT
			for j := range stftFrame {
				magnitude := cmplx.Abs(spectrogram[i][j])
				phase := cmplx.Phase(stftFrame[j])
				spectrogram[i][j] = cmplx.Rect(magnitude, phase)
			}
		}

		// Reconstruct the signal from the updated spectrogram
		reconstructedSignal = make([]float64, frameLen+(numFrames-1)*frameShift)
		windowSum = make([]float64, frameLen+(numFrames-1)*frameShift)
		for i := 0; i < numFrames; i++ {
			buf := fft.IFFT(spectrogram[i])
			index := 0
			for t := i * frameShift; t < i*frameShift+frameLen; t++ {
				reconstructedSignal[t] += real(buf[index]) * s.Window[index]
				windowSum[t] += s.Window[index]
				index++
			}
		}

		// Normalize reconstructed signal by window sum
		for i := range reconstructedSignal {
			if windowSum[i] != 0 {
				reconstructedSignal[i] /= windowSum[i]
			}
		}
	}

	return reconstructedSignal
}

// FromMel generates a wave buffer from a mel spectrogram and returns the wave buffer.
func (m *Mel) FromMel(ospectrum [][2]float64) ([]float64, error) {
	spectral_denormalize(ospectrum)

	stft1 := stft.New(m.Window, m.Resolut)

	undo := m.undospectrum(undomel(m.Resolut/2, m.NumMels, ospectrum, m.MelFmin, m.MelFmax))

	buf1 := ISTFT(stft1, undo, m.GriffinLimIterations)

	return buf1, nil
}

// LoadFlac loads mono flac file to sample vector
func LoadFlac(inputFile string) []float64 {
	mono, _ := loadflac(inputFile)
	return mono
}

// LoadWav loads mono wav file to sample vector
func LoadWav(inputFile string) []float64 {
	mono, _ := loadwav(inputFile)
	return mono
}

// SaveWav saves mono wav file from sample vector
func SaveWav(outputFile string, vec []float64, sr int) error {
	return dumpwav(outputFile, vec, sr)
}

func (m *Mel) Image(buf [][2]float64) []uint16 {
	return dumpbuffer(buf, m.NumMels)
}

// ToMel generates a mel spectrogram from an input FLAC audio file and saves it as a PNG image.
func (m *Mel) ToMelFlac(inputFile, outputFile string) error {

	var buf, sr = loadflac(inputFile)
	if len(buf) == 0 {
		return ErrFileNotLoaded
	}

	ospectrum, err := m.ToMel(buf)
	if err != nil {
		return err
	}

	dumpimage(outputFile, ospectrum, m.NumMels, m.YReverse, float64(len(buf)*m.NumMels)/float64(len(ospectrum)), float64(sr))

	return nil
}

// ToMel generates a mel spectrogram from an input WAV audio file and saves it as a PNG image.
func (m *Mel) ToMelWav(inputFile, outputFile string) error {

	var buf, sr = loadwav(inputFile)
	if len(buf) == 0 {
		return ErrFileNotLoaded
	}

	ospectrum, err := m.ToMel(buf)
	if err != nil {
		return err
	}

	dumpimage(outputFile, ospectrum, m.NumMels, m.YReverse, float64(len(buf)*m.NumMels)/float64(len(ospectrum)), float64(sr))

	return nil
}

func (m *Mel) ToWavPng(inputFile, outputFile string) error {

	var buf, samples, samplerate = loadpng(inputFile, m.YReverse)
	if len(buf) == 0 {
		return ErrFileNotLoaded
	}

	for i := range buf {
		buf[i][0] += m.VolumeBoost
		buf[i][1] += m.VolumeBoost
	}

	owave, err := m.FromMel(buf)
	if err != nil {
		return err
	}

	if int(samples) > 0 && isPadded(int(samples), len(owave), m.Window) && len(owave) > int(samples) {
		owave = owave[0:int(samples)]
	}
	if samplerate != 0 && m.SampleRate == 0 {
		m.SampleRate = int(samplerate)
	}

	dumpwav(outputFile, owave, m.SampleRate)

	return nil
}
