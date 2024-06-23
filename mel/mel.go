package libmel

import "github.com/r9y9/gossp/stft"
import "math"
import "errors"

// Mel represents the configuration for generating mel spectrograms.
type Mel struct {
	NumMels  int
	MelFmin  float64
	MelFmax  float64
	TuneMul  float64
	TuneAdd  float64
	Window	 int
	Resolut  int
	YReverse bool
}

// NewMel creates a new Mel instance with default values.
func NewMel() *Mel {
	return &Mel{
		NumMels: 160,
		MelFmin: 0,
		MelFmax: 8000,
		TuneMul: 0.0001,
		TuneAdd: 0.0001,
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

			//fmt.Println(i,j,realn)

			ospectrum = append(ospectrum, [2]float64{realn0, realn1})

		}
	}

	ospectrum = domel(m.Resolut/2, m.NumMels, ospectrum, m.MelFmin, m.MelFmax)

	spectral_normalize(ospectrum)

	return ospectrum, nil

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
