package image

import (
	"fmt"
	"github.com/fxsjy/gonn/gonn"
	"neural_network/vector"
	"testing"
)

func TestNNImages(t *testing.T) {
	CreateNN()
	// Загружем НС из файла.
	nn := gonn.LoadNN("imgs")
	inputFile := "B"
	income := vector.GetVectorFromImage("Images/B.jpg")

	// Получаем ответ от НС (массив весов)
	out := nn.Forward(income)
	fmt.Println(out)

	// Печатаем ответ на экран.
	fmt.Println(GetResult(out, inputFile))
}
func GetResult(output []float64, inputFile string) string {
	lenOut := len(output)
	i := 0.0
	for j := range output {
		i += output[j]
	}
	f := float64(lenOut)
	result := i * 100 / f
	return fmt.Sprint(inputFile, " = ", result, "%")
}

func CreateNN() {
	// Получаем массив входных данных

	x1 := vector.GetVectorFromImage("Images/O1.jpg")
	x2 := vector.GetVectorFromImage("Images/O2.jpg")
	x3 := vector.GetVectorFromImage("Images/O3.jpg")
	x4 := vector.GetVectorFromImage("Images/O4.jpg")
	x5 := vector.GetVectorFromImage("Images/O5.jpg")

	vectorArrays := [][]float64{x1, x2, x3, x4, x5}

	// Создаём массив входящих параметров:

	// Теперь создаём "цели" - те результаты, которые нужно получить

	x1 = vector.GetVectorFromImage("Images/B.jpg")
	x2 = vector.GetVectorFromImage("Images/B.jpg")
	x3 = vector.GetVectorFromImage("Images/B.jpg")

	target := [][]float64{vector.GetVectorFromImage("Images/O1.jpg"), vector.GetVectorFromImage("Images/O1.jpg"), vector.GetVectorFromImage("Images/O1.jpg"), vector.GetVectorFromImage("Images/O1.jpg"), vector.GetVectorFromImage("Images/O1.jpg"), vector.GetVectorFromImage("Images/O1.jpg")}

	// Создаём НС с 3 входными нейронами (столько же входных параметров),
	// 32 скрытыми нейронами и
	// 4 выходными нейронами (столько же вариантов ответа)
	nn := gonn.DefaultNetwork(len(x1), 32, len(x1), false)

	// Начинаем обучать нашу НС.
	// Количество итераций - 100000
	nn.Train(vectorArrays, target, 1000)

	// Сохраняем готовую НС в файл.
	gonn.DumpNN("imgs", nn)
}
