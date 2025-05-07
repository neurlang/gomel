package phase

import "github.com/r9y9/gossp/stft"
import "github.com/mjibson/go-dsp/fft"
import "errors"

// Phase represents the configuration for generating phase-preserving spectrograms.
type Phase struct {
	NumFreqs int
	Window   int
	Resolut  int
	YReverse bool
	// sample rate for output wav
	SampleRate  int
	VolumeBoost float64
}

// NewPhase creates a new Phase instance with default values.
func NewPhase() *Phase {
	return &Phase{
		NumFreqs:    768,
		Window:      1280,
		Resolut:     4096,
		VolumeBoost: 0,
	}
}

var ErrFileNotLoaded = errors.New("wavNotLoaded")

// ToPhase generates a phase-preserving spectrogram from a wave buffer and returns the spectrogram buffer.
func (m *Phase) ToPhase(buf []float64) ([][3]float64, error) {

	buf = pad(buf, m.Window)

	stft := stft.New(m.Window, m.Resolut)

	spectrum := stft.STFT(buf)
	var ospectrum [][3]float64

	for i := range spectrum {
		for j := 0; j < m.Resolut/2; j++ {

			var v0 = spectrum[i][j+1]
			var v1 = spectrum[i][m.Resolut-j-1]

			var realn1 = imag(v0)

			var realm0 = real(v1)
			var realm1 = imag(v1)

			ospectrum = append(ospectrum, [3]float64{realn1, realm0, realm1})

		}
	}

	ospectrum = shrink(ospectrum, m.Resolut/2, m.NumFreqs)
	spectral_normalize(ospectrum)

	return ospectrum, nil

}

func (m *Phase) undospectrum(ospectrum [][3]float64) (spectrum [][]complex128) {
	spectrum = make([][]complex128, len(ospectrum)/(m.Resolut/2))

	for i := range spectrum {
		spectrum[i] = make([]complex128, m.Resolut)
		for j := 0; j < m.Resolut/2; j++ {
			index := i*(m.Resolut/2) + j
			realn1 := ospectrum[index][0]
			realm0 := ospectrum[index][1]
			realm1 := ospectrum[index][2]

			v0 := complex(realm0, realn1) // cos
			v1 := complex(realm0, realm1) // sin

			spectrum[i][j+1] = v0
			spectrum[i][m.Resolut-j-1] = v1
		}
	}
	return
}

func ISTFT(s *stft.STFT, spectrogram [][]complex128, numIterations int) []float64 {
	frameShift := s.FrameShift
	frameLen := len(spectrogram[0])
	numFrames := len(spectrogram)
	reconstructedSignal := make([]float64, frameLen+(numFrames-1)*frameShift)

	for i := 0; i < numFrames; i++ {
		buf := fft.IFFT(spectrogram[i])
		for j := 0; j < frameLen; j++ {
			pos := i*frameShift + j
			if pos < len(reconstructedSignal) {
				val := real(buf[j]) * s.Window[j]
				reconstructedSignal[pos] += val
				//windowSum[pos] += s.Window[j] * s.Window[j] // Sum of squares
			}
		}
	}

	return reconstructedSignal
}

// FromPhase generates a wave buffer from a Phase spectrogram and returns the wave buffer.
func (m *Phase) FromPhase(ospectrum [][3]float64) ([]float64, error) {

	stft1 := stft.New(m.Window, m.Resolut)

	spectral_denormalize(ospectrum)
	ospectrum = grow(ospectrum, m.NumFreqs, m.Resolut/2)

	undo := m.undospectrum(ospectrum)

	buf1 := ISTFT(stft1, undo, 0)

	if m.VolumeBoost != 0 {
		for i := range buf1 {
			buf1[i] *= m.VolumeBoost
		}
	}

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

// LoadFlacSampleRate loads mono flac file to sample vector and it's sample rate, or it returns an error like ErrFileNotLoaded
func LoadFlacSampleRate(inputFile string) ([]float64, uint32, error) {
	mono, sr := loadflac(inputFile)
	if len(mono) == 0 || sr == 0 {
		return nil, 0, ErrFileNotLoaded
	}
	return mono, uint32(sr), nil
}

// LoadWavSampleRate loads mono wav file to sample vector and it's sample rate, or it returns an error like ErrFileNotLoaded
func LoadWavSampleRate(inputFile string) ([]float64, uint32, error) {
	mono, sr := loadwav(inputFile)
	if len(mono) == 0 || sr == 0 {
		return nil, 0, ErrFileNotLoaded
	}
	return mono, uint32(sr), nil
}

// SaveWav saves mono wav file from sample vector
func SaveWav(outputFile string, vec []float64, sr int) error {
	return dumpwav(outputFile, vec, sr)
}

func (m *Phase) Image(buf [][3]float64) []uint16 {
	return dumpbuffer(buf, m.NumFreqs)
}

// ToPhaseFlac generates a Phase spectrogram from an input FLAC audio file and saves it as a PNG image.
func (m *Phase) ToPhaseFlac(inputFile, outputFile string) error {

	var buf, sr = loadflac(inputFile)
	if len(buf) == 0 {
		return ErrFileNotLoaded
	}

	ospectrum, err := m.ToPhase(buf)
	if err != nil {
		return err
	}

	dumpimage(outputFile, ospectrum, m.NumFreqs, m.YReverse, float64(len(buf)*m.NumFreqs)/float64(len(ospectrum)), float64(sr))

	return nil
}

// ToPhaseWav generates a Phase spectrogram from an input WAV audio file and saves it as a PNG image.
func (m *Phase) ToPhaseWav(inputFile, outputFile string) error {

	var buf, sr = loadwav(inputFile)
	if len(buf) == 0 {
		return ErrFileNotLoaded
	}

	ospectrum, err := m.ToPhase(buf)
	if err != nil {
		return err
	}

	dumpimage(outputFile, ospectrum, m.NumFreqs, m.YReverse, float64(len(buf)*m.NumFreqs)/float64(len(ospectrum)), float64(sr))

	return nil
}

func (m *Phase) ToWavPng(inputFile, outputFile string) error {

	var buf, samples, samplerate = loadpng(inputFile, m.YReverse)
	if len(buf) == 0 {
		return ErrFileNotLoaded
	}

	owave, err := m.FromPhase(buf)
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
