package vector

import (
	"fmt"
	"github.com/oelmekki/matrix"
	"image"
	"image/color"
	_ "image/jpeg"
	"image/png"
	"log"
	"math"
	"os"
)

func VectorToMatrix(v1 []float64, v2 []float64) matrix.Matrix {

	m := matrix.GenerateMatrix(len(v1), len(v2))

	for i, elem := range v1 {
		for i2, elem2 := range v1 {
			m.SetAt(i, i2, elem2*elem)
		}
	}

	return m
}

func Sigmod(v float64) float64 {

	if v >= 0 {

		return 1

	} else {

		return -1
	}

}

func GetVectorFromImage(path string) ([]float64, error) {

	reader, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}
	defer reader.Close()
	m, _, err := image.Decode(reader)
	if err != nil {
		return nil, err
	}

	v := make([]float64, m.Bounds().Dx()*m.Bounds().Dy())
	vectorIteration := 0
	for x := 0; x < m.Bounds().Dx(); x++ {
		for y := 0; y < m.Bounds().Dy(); y++ {
			r, g, b, _ := m.At(x, y).RGBA()
			/*normalVal := float64(r+g+b) / 3 / 257
			if normalVal >= 150 {
				v[vectorIteration] = 1
			} else {
				v[vectorIteration] = -1
			}*/
			v[vectorIteration] = float64(r + g + b)
			vectorIteration++
		}
	}
	return v, nil
}

func NNVector() {

	fmt.Println("Memory size ~ ", int(400/(2*math.Log2(400))), " objects")
	fmt.Println("1 - А")
	fmt.Println("2 - Б")
	fmt.Println("3 - О")
	fmt.Println("-----Start------")

	vectorArrays := [3][]float64{}

	x1, _ := GetVectorFromImage("Images/A.jpg")
	x2, _ := GetVectorFromImage("Images/B.jpg")
	x3, _ := GetVectorFromImage("Images/O.jpg")

	y, _ := GetVectorFromImage("Images/Income.jpg")

	vectorArrays[0] = x1
	vectorArrays[1] = x2
	vectorArrays[2] = x3

	matrixArray := [len(vectorArrays)]matrix.Matrix{}

	for i, vInArray := range vectorArrays {

		matrixArray[i] = VectorToMatrix(vInArray, vInArray)
	}

	W := matrix.Matrix{}

	for i, matrixInArray := range matrixArray {
		if i == 0 {
			W = matrixInArray
			continue
		}
		W, _ = W.Add(matrixInArray)
	}

	for i := 0; i < W.Rows(); i++ {
		W.SetAt(i, i, 0)
	}

	S := make([]float64, 400)

	for II := 0; II < 100; II++ {
		if II == 0 {
			S, _ = W.VectorMultiply(y)
			for i, element := range S {

				S[i] = Sigmod(element)
			}
			continue

		} else {

			S, _ = W.VectorMultiply(S)
			for i, element := range S {

				S[i] = Sigmod(element)
			}
		}

	}

	ar := [3]int{1, 1, 1}

	for vectorI, v := range vectorArrays {
		for i, elem := range v {
			if elem != S[i] {
				ar[vectorI] = 0
				break
			}
		}
	}

	for i, el := range ar {
		if el == 1 {
			fmt.Println("Looks like", i+1)
		}
	}

	img := image.NewRGBA(image.Rect(0, 0, 20, 20))
	xx := 0
	yy := 0

	for i := 0; i < 400; i++ {
		if i%20 == 0 {
			yy++
			xx = 0
		} else {
			xx++
		}
		if S[i] == -1 {
			img.Set(xx, yy, color.RGBA{0, 0, 0, 255})
		} else {
			img.Set(xx, yy, color.RGBA{255, 255, 255, 255})
		}

	}

	f, _ := os.OpenFile("Images/out.png", os.O_WRONLY|os.O_CREATE, 0600)
	png.Encode(f, img)
	f.Close()
	var str string
	fmt.Scanln(&str)
}
