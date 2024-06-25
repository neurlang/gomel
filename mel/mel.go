package mel

import "github.com/r9y9/gossp/stft"
import "github.com/mjibson/go-dsp/fft"
import "math"
import "errors"
import "math/cmplx"

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

	// spread when loading spectrogram from image, can be a value like -10
	Spread int
}

// NewMel creates a new Mel instance with default values.
func NewMel() *Mel {
	return &Mel{
		NumMels: 160,
		MelFmin: 0,
		MelFmax: 8000,
		TuneMul: 1,
		TuneAdd: 0,
		Window:  256,
		Resolut: 2048,
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

			var realn0 = math.Sqrt(real(v0)*real(v0)+imag(v0)*imag(v0))*float64(m.TuneMul) + m.TuneAdd

			var v1 = spectrum[i][m.Resolut-j-1]

			var realn1 = math.Sqrt(real(v1)*real(v1)+imag(v1)*imag(v1))*float64(m.TuneMul) + m.TuneAdd

			ospectrum = append(ospectrum, [2]float64{realn0, realn1})

		}
	}

	ospectrum = domel(m.Resolut/2, m.NumMels, ospectrum, m.MelFmin, m.MelFmax)

	spectral_normalize(ospectrum)

	return ospectrum, nil

}

func ISTFT(s *stft.STFT, spectrogram [][]complex128, numIterations int) []float64 {
	frameLen := len(spectrogram[0])
	numFrames := len(spectrogram)
	reconstructedSignal := make([]float64, frameLen+(numFrames-1)*s.FrameShift)
	windowSum := make([]float64, frameLen+(numFrames-1)*s.FrameShift)

	// Initial reconstruction
	for i := 0; i < numFrames; i++ {
		buf := fft.IFFT(spectrogram[i])
		index := 0
		for t := i * s.FrameShift; t < i*s.FrameShift+frameLen; t++ {
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

	// Griffin-Lim iterations
	for iter := 0; iter < numIterations; iter++ {
		// Calculate the STFT of the reconstructed signal
		for i := 0; i < numFrames; i++ {
			frame := make([]float64, frameLen)
			for j := 0; j < frameLen; j++ {
				if i*s.FrameShift+j < len(reconstructedSignal) {
					frame[j] = reconstructedSignal[i*s.FrameShift+j] * s.Window[j]
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
		reconstructedSignal = make([]float64, frameLen+(numFrames-1)*s.FrameShift)
		windowSum = make([]float64, frameLen+(numFrames-1)*s.FrameShift)
		for i := 0; i < numFrames; i++ {
			buf := fft.IFFT(spectrogram[i])
			index := 0
			for t := i * s.FrameShift; t < i*s.FrameShift+frameLen; t++ {
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

	ospectrum = undomel(m.Resolut/2, m.NumMels, ospectrum, m.MelFmin, m.MelFmax)

	for r := 0; r < int(math.Sqrt(float64(m.MelFmax-m.MelFmin)/float64(m.NumMels))); r++ {
		for l := 0; l < 2; l++ {
			for x := 0; x < len(ospectrum)/(m.Resolut/2); x++ {
				for y := 1; y+1 < m.Resolut/2; y++ {
					ospectrum[y+x*(m.Resolut/2)][l] = (ospectrum[y-1+x*(m.Resolut/2)][l] +
						ospectrum[y+0+x*(m.Resolut/2)][l] +
						ospectrum[y+1+x*(m.Resolut/2)][l]) / 3
				}
			}
		}
	}

	spectrum := m.undospectrum(ospectrum)

	stft := stft.New(m.Window, m.Resolut)

	buf := ISTFT(stft, spectrum, m.GriffinLimIterations)

	return buf, nil
}

// LoadFlac loads mono flac file to sample vector
func LoadFlac(inputFile string) []float64 {
	return loadflac(inputFile)
}

// LoadWav loads mono wav file to sample vector
func LoadWav(inputFile string) []float64 {
	return loadwav(inputFile)
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

	var buf = loadflac(inputFile)
	if len(buf) == 0 {
		return ErrFileNotLoaded
	}

	ospectrum, err := m.ToMel(buf)
	if err != nil {
		return err
	}

	dumpimage(outputFile, ospectrum, m.NumMels, m.YReverse)

	return nil
}

// ToMel generates a mel spectrogram from an input WAV audio file and saves it as a PNG image.
func (m *Mel) ToMelWav(inputFile, outputFile string) error {

	var buf = loadwav(inputFile)
	if len(buf) == 0 {
		return ErrFileNotLoaded
	}

	ospectrum, err := m.ToMel(buf)
	if err != nil {
		return err
	}

	dumpimage(outputFile, ospectrum, m.NumMels, m.YReverse)

	return nil
}

func (m *Mel) ToWavPng(inputFile, outputFile string) error {

	var buf = loadpng(inputFile, m.YReverse, m.Spread)
	if len(buf) == 0 {
		return ErrFileNotLoaded
	}

	owave, err := m.FromMel(buf)
	if err != nil {
		return err
	}

	dumpwav(outputFile, owave, 44100)

	return nil
}
