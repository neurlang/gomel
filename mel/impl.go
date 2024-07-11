package mel

import "image"
import "image/png"
import "os"
import "io"
import "image/color"
import "github.com/faiface/beep"
import "github.com/faiface/beep/wav"
import "github.com/mewkiz/flac"
import "math"
import "math/cmplx"
import "encoding/binary"

func dumpbuffer(buf [][2]float64, mels int) (out []uint16) {
	stride := len(buf) / mels

	var mgc_max, mgc_min = [2]float64{(-99999999.), (-99999999.)}, [2]float64{(9999999.), (9999999.)}

	for l := 0; l < 2; l++ {
		for x := 0; x < stride; x++ {
			for y := 0; y < mels; y++ {
				var w = buf[stride*y+x][l]
				if w > mgc_max[l] {
					mgc_max[l] = w
				}
				if w < mgc_min[l] {
					mgc_min[l] = w
				}
			}
		}
	}
	for y := 0; y < mels; y++ {
		for x := 0; x < stride; x++ {
			val0 := (buf[stride*y+x][0] - mgc_min[0]) / (mgc_max[0] - mgc_min[0])
			val1 := (buf[stride*y+x][1] - mgc_min[1]) / (mgc_max[1] - mgc_min[1])
			val := uint16(int(255*val0)) | uint16(int(255*val1))<<8

			out = append(out, val)
		}
	}
	return
}

func unpackBytesToFloat64(bytes []byte) float64 {
	bits := binary.LittleEndian.Uint64(bytes) // Read the bits from the byte slice
	f := math.Float64frombits(bits)           // Convert uint64 bits to float64
	return f
}

func loadpng(name string, reverse bool) (buf [][2]float64) {
	// Open the PNG file
	file, err := os.Open(name)
	if err != nil {
		println(err.Error())
		return nil
	}
	defer file.Close()

	// Decode the PNG file
	img, err := png.Decode(file)
	if err != nil {
		println(err.Error())
		return nil
	}

	// Get the bounds of the image
	bounds := img.Bounds()
	var floats []byte
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {

			var color color.Color
			if reverse {
				// Get the color of the pixel at (x, y) with y-coordinate reversed
				color = img.At(x, bounds.Max.Y-y-1)
			} else {
				// Get the color of the pixel at (x, y)
				color = img.At(x, y)
			}
			r, g, b, _ := color.RGBA()

			if x == 0 && y < 16 {
				floats = append(floats, byte(b>>8))
			}

			val0 := float64(r>>8) / 255
			val1 := float64(g>>8) / 255

			val := [2]float64{val0, val1}

			buf = append(buf, val)
		}
	}
	var mgc_max, mgc_min = unpackBytesToFloat64(floats[0:8]), unpackBytesToFloat64(floats[8:16])

	for i := range buf {
		buf[i][0] = (buf[i][0]*(mgc_max-mgc_min) + mgc_min)
		buf[i][1] = (buf[i][1]*(mgc_max-mgc_min) + mgc_min)
	}
	//dumpimage("test.png", buf, 160, reverse)
	return
}

func packFloat64ToBytes(f float64) []byte {
	bits := math.Float64bits(f)                // Convert float64 to uint64
	bytes := make([]byte, 8)                   // Create a byte slice of size 8
	binary.LittleEndian.PutUint64(bytes, bits) // Write the bits to the byte slice in little-endian order
	return bytes
}

func dumpimage(name string, buf [][2]float64, mels int, reverse bool) error {

	f, err := os.Create(name)
	if err != nil {
		return err
	}

	stride := len(buf) / mels

	img := image.NewNRGBA(image.Rect(0, 0, stride, mels))

	var mgc_max, mgc_min = (-math.MaxFloat64), (math.MaxFloat64)

	for x := 0; x < stride; x++ {
		for l := 0; l < 2; l++ {
			for y := 0; y < mels; y++ {
				var w = buf[stride*y+x][l]
				if w > mgc_max {
					mgc_max = w
				}
				if w < mgc_min {
					mgc_min = w
				}
			}
		}
	}
	floats := append(packFloat64ToBytes(mgc_max), packFloat64ToBytes(mgc_min)...)
	//println(mgc_max, mgc_min)
	for x := 0; x < stride; x++ {
		for y := 0; y < mels; y++ {
			var col color.NRGBA
			val0 := (buf[stride*y+x][0] - mgc_min) / (mgc_max - mgc_min)
			val1 := (buf[stride*y+x][1] - mgc_min) / (mgc_max - mgc_min)

			col.R = uint8(int(255 * val0))
			col.G = uint8(int(255 * val1))
			if x == 0 {
				col.B = uint8(int(floats[y&15]))
			} else {
				col.B = 0
			}
			col.A = uint8(255)
			if reverse {
				img.SetNRGBA(x, mels-y-1, col)
			} else {
				img.SetNRGBA(x, y, col)
			}
		}
	}

	if err := png.Encode(f, img); err != nil {
		f.Close()
		return err
	}

	if err := f.Close(); err != nil {
		return err
	}

	return nil
}

func dumpwav(name string, data []float64, sr int) error {
	noise := beep.StreamerFunc(func(samples [][2]float64) (n int, ok bool) {
		if len(data) < len(samples) {
			for i := range data {
				samples[i][0] = data[i]
				samples[i][1] = data[i]
			}
			l := len(data)
			data = nil
			return l, false
		} else {
			for i := range samples {
				samples[i][0] = data[i]
				samples[i][1] = data[i]
			}
			data = data[len(samples):]
			return len(samples), true
		}
	})

	// Create an output file
	f, err := os.Create(name)
	if err != nil {
		return err
	}
	defer f.Close()

	// Create a WAV encoder
	err = wav.Encode(f, noise, beep.Format{
		SampleRate:  beep.SampleRate(sr),
		NumChannels: 1,
		Precision:   2,
	})
	if err != nil {
		return err
	}
	return nil
}

func loadwav(name string) (out []float64) {
	file, _ := os.Open(name)
	defer file.Close()

	// wavReader
	stream, _, err := wav.Decode(file)
	if err != nil {
		println(err.Error())
		return nil
	}

	// require wavReader
	if stream == nil {
		return nil
	}
	var samples = make([][2]float64, 0, 1)
	for {
		samples = samples[0:1]
		n, ok := stream.Stream(samples)
		if !ok {
			break
		}
		samples = samples[:n]
		for i := 0; i < 1; i++ {
			out = append(out, samples[i][0])
		}
	}
	return
}

func loadflac(name string) (out []float64) {
	// Open love.flac for audio streaming without parsing metadata.
	stream, err := flac.Open(name)
	if err != nil {
		println(err.Error())
		return nil
	}
	defer stream.Close()

	for {
		// Parse one frame of audio samples at the time, each frame containing one
		// subframe per audio channel.
		frame, err := stream.ParseNext()
		if err != nil {
			if err == io.EOF {
				break
			}
		}

		for _, subframe := range frame.Subframes {
			for _, sample := range subframe.Samples {
				out = append(out, float64(sample)/(256*256))
			}
			//break
		}
	}
	return
}

func mel_to_hz(value float64) float64 {
	const _MEL_BREAK_FREQUENCY_HERTZ = 700.0
	const _MEL_HIGH_FREQUENCY_Q = 1127.0
	return _MEL_BREAK_FREQUENCY_HERTZ * (math.Exp(value/_MEL_HIGH_FREQUENCY_Q) - 1.0)
}

func hz_to_mel(value float64) float64 {
	const _MEL_BREAK_FREQUENCY_HERTZ = 700.0
	const _MEL_HIGH_FREQUENCY_Q = 1127.0
	return _MEL_HIGH_FREQUENCY_Q * math.Log(1.0+(value/_MEL_BREAK_FREQUENCY_HERTZ))
}

func domel(filtersize, mels int, spectrum [][2]float64, mel_fmin, mel_fmax float64) (melspectrum [][2]float64) {
	melbin := hz_to_mel(mel_fmax) / float64(mels)

	for i := 0; i < mels; i++ {
		for j := 0; j < len(spectrum); j += filtersize {
			vallo := float64(filtersize) * (mel_fmin + mel_to_hz(melbin*float64(i))) / (mel_fmax + mel_fmin)
			valhi := float64(filtersize) * (mel_fmin + mel_to_hz(melbin*float64(i+1))) / (mel_fmax + mel_fmin)

			inlo, modlo := math.Modf(vallo)
			inhi := math.Floor(valhi)
			if inlo < 0 {
				inlo, modlo, inhi = 0, 0, 0
			}

			var tot [2]float64
			for l := 0; l < 2; l++ {
				var total float64

				if int(inlo)+1 == int(inhi) {
					total += spectrum[j+int(inlo)][l] * (1 - modlo)
					total += spectrum[j+int(inhi)][l] * modlo
				} else {
					for k := int(inlo); k < int(inhi); k++ {
						total += spectrum[j+k][l]
					}
					total /= float64(int(inhi) - int(inlo) + 1)
				}

				tot[l] = total
			}
			melspectrum = append(melspectrum, tot)
		}
	}

	return
}

func undomel(filtersize, mels int, melspectrum [][2]float64, mel_fmin, mel_fmax float64) (spectrum [][2]float64) {
	filterbin := hz_to_mel(mel_fmax) / float64(mels)
	stride := len(melspectrum) / mels

	for j := 0; j < stride; j++ {
		for i := 0; i < filtersize; i++ {
			vallo := float64(hz_to_mel((float64(i)*(mel_fmax+mel_fmin)/float64(filtersize))-mel_fmin) / filterbin)
			valhi := float64(hz_to_mel((float64(i+1)*(mel_fmax+mel_fmin)/float64(filtersize))-mel_fmin) / filterbin)

			inlo, modlo := math.Modf(vallo)
			inhi := math.Floor(valhi)
			if inlo < 0 {
				inlo, modlo, inhi = 0, 0, 0
			}

			var tot [2]float64
			for l := 0; l < 2; l++ {
				var total float64

				if int(inlo) == int(inhi) {
					total += melspectrum[j+stride*int(inlo)][l]
				} else if int(inlo)+1 == int(inhi) && int(inhi) < mels {
					total += melspectrum[j+stride*int(inlo)][l] * (1 - modlo)
					total += melspectrum[j+stride*int(inhi)][l] * modlo
				} else {
					for k := int(inlo); k < int(inhi); k++ {
						total += melspectrum[j+stride*k][l]
					}
					total /= inhi - inlo + 1
				}

				tot[l] += total
			}
			spectrum = append(spectrum, tot)
		}
	}

	return
}

func (m *Mel) undospectrum(ospectrum [][2]float64) (spectrum [][]complex128) {
	spectrum = make([][]complex128, len(ospectrum)/(m.Resolut/2))

	for i := range spectrum {
		spectrum[i] = make([]complex128, m.Resolut)
		for j := 0; j < m.Resolut/2; j++ {
			index := i*(m.Resolut/2) + j
			realn0 := ospectrum[index][0]
			realn1 := ospectrum[index][1]

			real0 := (realn0 - m.TuneAdd) / m.TuneMul
			real1 := (realn1 - m.TuneAdd) / m.TuneMul

			v0 := cmplx.Rect(real0, 0)
			v1 := cmplx.Rect(real1, 0)

			spectrum[i][j] = v0
			spectrum[i][m.Resolut-j-1] = v1
		}
	}

	return
}

func spectral_normalize(buf [][2]float64) {
	for l := 0; l < 2; l++ {
		for i := range buf {
			if buf[i][l] < 1e-5 {
				buf[i][l] = 1e-5
			}
			buf[i][l] = float64(math.Log(float64(buf[i][l])))
		}
	}
}

func spectral_denormalize(buf [][2]float64) {
	for l := 0; l < 2; l++ {
		for i := range buf {
			buf[i][l] = float64(math.Exp(float64(buf[i][l])))
		}
	}
}

func pad(buf []float64, filter int) []float64 {
	// Calculate the current length of the buffer
	currentLen := len(buf)

	// Calculate the minimum target size (15 * filter)
	minTargetSize := 15 * filter

	// Calculate the required padding length
	padLen := 0
	if currentLen >= minTargetSize {
		// Find the next target size which is a multiple of filter greater than currentLen
		remainder := (currentLen - minTargetSize) % filter
		if remainder != 0 {
			padLen = filter - remainder - 1
		}
	} else {
		// If currentLen is less than the minimum target size, pad to minTargetSize
		padLen = minTargetSize - currentLen - 1
	}

	// Append the necessary padding
	if padLen > 0 {
		buf = append(buf, make([]float64, padLen)...)
	}

	return buf
}
