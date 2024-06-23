package libmel

import "image"
import "image/png"
import "os"
import "image/color"
import "github.com/faiface/beep/wav"
import "math"

func dumpimage(name string, buf [][2]float64, mels int, reverse bool) error {
	f, err := os.Create(name)
	if err != nil {
		return err
	}

	stride := len(buf) / mels

	img := image.NewRGBA(image.Rect(0, 0, stride, mels))

	var mgc_max, mgc_min = [2]float64{(-99999999.),(-99999999.)}, [2]float64{(9999999.),(9999999.)}

	for x := 0; x < stride; x++ {
		for l := 0; l < 2; l++ {
		for y := 0; y < mels; y++ {
			var w = buf[stride*y+x][l]
			if w > mgc_max[l] {
				mgc_max[l] = w
			}
			if w < mgc_min[l] {
				mgc_min[l] = w
			}
		}}
	}
	for x := 0; x < stride; x++ {
		for y := 0; y < mels; y++ {
			var col color.RGBA
			val0 := (buf[stride*y+x][0] - mgc_min[0]) / (mgc_max[0] - mgc_min[0])
			val1 := (buf[stride*y+x][1] - mgc_min[1]) / (mgc_max[1] - mgc_min[1])
			col.R = uint8(int(255 * val0))
			col.G = uint8(int(255 * val1))
			col.B = uint8(int(255 * (val0+val1)*0.5))
			col.A = uint8(255)
			if reverse {
				img.SetRGBA(x, mels-y-1, col)
			} else {
				img.SetRGBA(x, y, col)
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

	var melbin = hz_to_mel(mel_fmax) / float64(mels)

	for i := 0; i < mels; i++ {
		//var j = 0
		for j := 0; j < len(spectrum); j += filtersize {

			var vallo = float64(filtersize) * (mel_fmin + mel_to_hz(melbin*float64(i+0))) / (mel_fmax + mel_fmin)
			var valhi = float64(filtersize) * (mel_fmin + mel_to_hz(melbin*float64(i+1))) / (mel_fmax + mel_fmin)

			var inlo, modlo = math.Modf(vallo)
			var inhi = math.Floor(valhi)
			if inlo < 0 {
				inlo, modlo, inhi = 0, 0, 0
			}
			var tot [2]float64
			for l := 0; l< 2; l++ {

				var total float64

				if int(inlo)+1 == int(inhi) {
					total += spectrum[j+int(inlo)][l] * float64(1-modlo)
					total += spectrum[j+int(inhi)][l] * float64(modlo)
				} else {

					for k := int(inlo); k < int(inhi); k++ {
						var sample = spectrum[j+k][l]
						total += sample
					}
				}

				total /= float64(int(inhi) - int(inlo) + 1)

				tot[l] = total
			}
			melspectrum = append(melspectrum, tot)

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

func pad(buf []float64, filter int) []float64 {
	if len(buf)&1 == 1 {
		buf = append(buf, 0)
	}
	if len(buf)&2 == 2 {
		buf = append([]float64{0}, buf...)
		buf = append(buf, 0)
	}
	if len(buf)&4 == 4 {
		buf = append([]float64{0, 0}, buf...)
		buf = append(buf, 0, 0)
	}
	for len(buf)%filter != 0 {
		buf = append([]float64{0, 0, 0, 0}, buf...)
		buf = append(buf, 0, 0, 0, 0)
	}
	for i := 0; i < filter/4; i++ {
		buf = append([]float64{0, 0, 0, 0}, buf...)
		buf = append(buf, 0, 0, 0, 0)
	}
	return buf
}


