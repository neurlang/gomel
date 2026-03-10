package phase

import "image"
import "image/png"
import "os"
import "io"
import "image/color"
import "github.com/faiface/beep"
import "github.com/faiface/beep/wav"
import "github.com/mewkiz/flac"
import "github.com/x448/float16"
import "math"
import "encoding/binary"

func dumpbuffer(buf [][2]float64, mels int) (out []uint16) {
	stride := len(buf) / mels

	var mgc_max, mgc_min = [2]float64{(-99999999.), (-99999999.)}, [2]float64{(9999999.), (9999999.)}

	for l := 0; l < 2; l++ {
		for x := 0; x < stride; x++ {
			for y := 0; y < mels; y++ {
				var w = buf[y+x*mels][l]
				if w > mgc_max[l] {
					mgc_max[l] = w
				}
				if w < mgc_min[l] {
					mgc_min[l] = w
				}
			}
		}
	}
	for x := 0; x < stride; x++ {
		for y := 0; y < mels; y++ {
			val0 := (buf[y+x*mels][0] - mgc_min[0]) / (mgc_max[0] - mgc_min[0])
			val1 := (buf[y+x*mels][1] - mgc_min[1]) / (mgc_max[1] - mgc_min[1])
			val := uint16(int(255*val0)) | uint16(int(255*val1))<<8

			out = append(out, val)
		}
	}
	return
}

func unpackBytesToFloat64(bytes []byte) float64 {
	bits := binary.LittleEndian.Uint16(bytes)      // Read the bits from the byte slice
	f := float64(float16.Frombits(bits).Float32()) // Convert uint64 bits to float64
	return f
}

func loadpng(name string, reverse bool, ihsPasses int, hdr bool) (buf [][2]float64, samples, samplerate float64) {
	// Open the PNG file
	file, err := os.Open(name)
	if err != nil {
		println(err.Error())
		return nil, 0, 0
	}
	defer file.Close()

	// Decode the PNG file
	img, err := png.Decode(file)
	if err != nil {
		println(err.Error())
		return nil, 0, 0
	}

	// Get the bounds of the image
	bounds := img.Bounds()
	mels := bounds.Max.Y - bounds.Min.Y
	maxVal := 65535
	if !hdr {
		maxVal = 255
	}
	var floats []byte
	for x := bounds.Min.X; x < bounds.Max.X; x++ {
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {

			var color color.Color
			if reverse {
				// Get the color of the pixel at (x, y) with y-coordinate reversed
				color = img.At(x, bounds.Max.Y-y-1)
			} else {
				// Get the color of the pixel at (x, y)
				color = img.At(x, y)
			}
			r, g, b, _ := color.RGBA()

			// Extract metadata from first column (x=0) blue channel at high-y end
			metaStart := mels - 16
			if x == 0 && y >= metaStart {
				if hdr {
					// For HDR, metadata bytes are stored as uint16 values (0-255 range)
					floats = append(floats, byte(b&0xFF))
				} else {
					floats = append(floats, byte(b>>8))
				}
			}

			var val0, val1 float64
			if hdr {
				// RGBA() returns 16-bit values (0-65535) for all image types
				val0 = float64(r) / float64(maxVal)
				val1 = float64(g) / float64(maxVal)
				// val2 (blue) not stored, will be reconstructed as -val0
			} else {
				// For 8-bit, RGBA() still returns 16-bit, so shift down
				val0 = float64(r>>8) / 255
				val1 = float64(g>>8) / 255
				// val2 (blue) not stored, will be reconstructed as -val0
			}

			val := [2]float64{val0, val1}

			buf = append(buf, val)
		}
	}
	var mgc_max0, mgc_max1, _, mgc_min0, mgc_min1, _, samples_in_mel, sr = unpackBytesToFloat64(floats[0:2]),
		unpackBytesToFloat64(floats[2:4]),
		unpackBytesToFloat64(floats[4:6]),
		unpackBytesToFloat64(floats[6:8]),
		unpackBytesToFloat64(floats[8:10]),
		unpackBytesToFloat64(floats[10:12]),
		unpackBytesToFloat64(floats[12:14]),
		unpackBytesToFloat64(floats[14:16])

	// Replace metadata pixels (blue channel, x=0, high-y rows) with the pixel just below
	// Note: Blue channel not stored in 2-channel format, so no replacement needed
	metaStart := mels - 16
	donorY := metaStart - 1
	if donorY < 0 {
		donorY = 0
	}
	// No-op: blue channel will be reconstructed as -red

	for i := range buf {
		buf[i][0] = (buf[i][0]*(mgc_max0-mgc_min0) + mgc_min0)
		buf[i][1] = (buf[i][1]*(mgc_max1-mgc_min1) + mgc_min1)
	}

	// Undo asinh compression
	for p := 0; p < ihsPasses; p++ {
		for i := range buf {
			for l := 0; l < 2; l++ {
				buf[i][l] = math.Sinh(buf[i][l])
			}
		}
	}

	samples = samples_in_mel * float64(bounds.Max.X-bounds.Min.X)
	samplerate = sr
	//dumpimage("test.png", buf, 160, reverse)
	return
}

func packFloat16ToBytes(f float64) []byte {
	var buf [2]byte
	bits := float16.Fromfloat32(float32(f)).Bits()
	binary.LittleEndian.PutUint16(buf[:], bits)
	return buf[:]
}
func joinBytes(b ...[]byte) (ret []byte) {
	for _, bytes := range b {
		ret = append(ret, bytes...)
	}
	return ret
}

func dumpimage(name string, buf [][2]float64, mels int, reverse bool, samples_in_mel, sr float64, ihsPasses int, hdr bool) error {

	// Apply asinh compression
	for p := 0; p < ihsPasses; p++ {
		for i := range buf {
			for l := 0; l < 2; l++ {
				buf[i][l] = math.Asinh(buf[i][l])
			}
		}
	}

	f, err := os.Create(name)
	if err != nil {
		return err
	}

	stride := len(buf) / mels

	maxVal := 65535
	if !hdr {
		maxVal = 255
	}

	var img image.Image
	if hdr {
		img = image.NewNRGBA64(image.Rect(0, 0, stride, mels))
	} else {
		img = image.NewNRGBA(image.Rect(0, 0, stride, mels))
	}

	var mgc_max, mgc_min = [2]float64{(-math.MaxFloat64), (-math.MaxFloat64)}, [2]float64{(math.MaxFloat64), (math.MaxFloat64)}

	for x := 0; x < stride; x++ {
		for l := 0; l < 2; l++ {
			for y := 0; y < mels; y++ {
				var w = buf[y+x*mels][l]
				if w > mgc_max[l] {
					mgc_max[l] = w
				}
				if w < mgc_min[l] {
					mgc_min[l] = w
				}
			}
		}
	}
	var floats = joinBytes(
		packFloat16ToBytes(mgc_max[0]),
		packFloat16ToBytes(mgc_max[1]),
		packFloat16ToBytes(0), // Placeholder for removed channel
		packFloat16ToBytes(mgc_min[0]),
		packFloat16ToBytes(mgc_min[1]),
		packFloat16ToBytes(0), // Placeholder for removed channel
		packFloat16ToBytes(samples_in_mel),
		packFloat16ToBytes(sr),
	)

	//println(mgc_max[0], mgc_min[0])
	for x := 0; x < stride; x++ {
		for y := 0; y < mels; y++ {
			val0 := (buf[y+x*mels][0] - mgc_min[0]) / (mgc_max[0] - mgc_min[0])
			val1 := (buf[y+x*mels][1] - mgc_min[1]) / (mgc_max[1] - mgc_min[1])
			val2 := -val0 // Blue channel reconstructed from conjugate symmetry

			metaStart := mels - len(floats)
			
			if hdr {
				var col color.NRGBA64
				col.R = uint16(int(float64(maxVal) * val0))
				col.G = uint16(int(float64(maxVal) * val1))
				if x == 0 && y >= metaStart {
					// Store metadata byte as uint16 (matching Python behavior)
					col.B = uint16(floats[y-metaStart])
				} else {
					col.B = uint16(int(float64(maxVal) * val2))
				}
				col.A = uint16(65535)
				if reverse {
					img.(*image.NRGBA64).SetNRGBA64(x, mels-y-1, col)
				} else {
					img.(*image.NRGBA64).SetNRGBA64(x, y, col)
				}
			} else {
				var col color.NRGBA
				col.R = uint8(int(float64(maxVal) * val0))
				col.G = uint8(int(float64(maxVal) * val1))
				if x == 0 && y >= metaStart {
					col.B = uint8(int(floats[y-metaStart]))
				} else {
					col.B = uint8(int(float64(maxVal) * val2))
				}
				col.A = uint8(255)
				if reverse {
					img.(*image.NRGBA).SetNRGBA(x, mels-y-1, col)
				} else {
					img.(*image.NRGBA).SetNRGBA(x, y, col)
				}
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

func loadwav(name string) (out []float64, sr float64) {
	file, _ := os.Open(name)
	defer file.Close()

	// wavReader
	stream, format, err := wav.Decode(file)
	if err != nil {
		println(err.Error())
		return nil, 0
	}

	sr = float64(format.SampleRate)

	// require wavReader
	if stream == nil {
		return nil, 0
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

func loadflac(name string) (out []float64, sr float64) {
	stream, err := flac.Open(name)
	if err != nil {
		println(err.Error())
		return nil, 0
	}
	defer stream.Close()

	// Iterate over the metadata blocks to find the StreamInfo block
	var sampleRate = stream.Info.SampleRate
	sr = float64(sampleRate)

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
				out = append(out, float64(sample)/(256*128))
			}
			//break
		}
	}
	return
}

func shrink(ospectrum [][2]float64, imels, omels int) (out [][2]float64) {
	for i := range ospectrum {
		j := i % imels
		if j < omels {
			out = append(out, ospectrum[i])
		}
	}
	return
}
func grow(ospectrum [][2]float64, imels, omels int) (out [][2]float64) {
	for i := range ospectrum {
		j := i % imels
		out = append(out, ospectrum[i])
		if j+1 == imels {
			for k := imels; k < omels; k++ {
				out = append(out, ospectrum[i])
			}
		}
	}
	return
}

func spectral_normalize(buf [][2]float64) {
	for l := 0; l < 2; l++ {
		for i := range buf {
			if buf[i][l] < 1e-10 {
				buf[i][l] = 1e-10
			}
			buf[i][l] = float64(math.Log2(float64(buf[i][l])))
		}
	}
}

func spectral_denormalize(buf [][2]float64) {
	for l := 0; l < 2; l++ {
		for i := range buf {
			buf[i][l] = float64(math.Exp2(float64(buf[i][l])))
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

func isPadded(originalLen, paddedLen, filter int) bool {
	// Calculate the minimum target size
	minTargetSize := 15 * filter

	if originalLen >= minTargetSize {
		// Calculate the remainder
		remainder := (originalLen - minTargetSize) % filter
		if remainder != 0 {
			// Calculate the required padding length
			padLen := filter - remainder - 1
			// Check if the padded length matches the expected padded length
			return paddedLen == originalLen+padLen
		} else {
			// No padding needed if remainder is zero
			return paddedLen == originalLen
		}
	} else {
		// Calculate the required padding length
		padLen := minTargetSize - originalLen - 1
		// Check if the padded length matches the expected padded length
		return paddedLen == originalLen+padLen
	}
}

func padShift(sampleRate int) (zeroPad, zeroShift int) {
	// 48000 family
	if sampleRate == 48000 {
		return 0, 0
	}
	if sampleRate == 32000 {
		return 2, 1 // 1.5x
	}
	if sampleRate == 24000 {
		return 1, 1 // 2x
	}
	if sampleRate == 16000 {
		return 1, 2 // 3x
	}
	if sampleRate == 8000 {
		return 1, 5 // 6x
	}
	// 44100 family
	if sampleRate == 44100 {
		return 0, 0
	}
	if sampleRate == 22050 {
		return 1, 1 // 2x
	}
	if sampleRate == 11025 {
		return 1, 3 // 4x
	}
	return 0, 0
}

func zeroStuffUpsample(audio []float64, zeroPad, zeroShift int) []float64 {
	if zeroPad == 0 {
		return audio
	}

	// Calculate output length
	numGroups := (len(audio) + zeroPad - 1) / zeroPad
	outputLen := len(audio) + numGroups*zeroShift
	output := make([]float64, outputLen)

	// Insert original samples with zeros in between
	outIdx := 0
	boost := float64(1 + zeroShift) // Compensate for energy loss
	for i := 0; i < len(audio); i++ {
		output[outIdx] = audio[i] * boost
		outIdx++
		// After every zeroPad samples, insert zeroShift zeros
		if (i+1)%zeroPad == 0 {
			outIdx += zeroShift
		}
	}

	return output
}
