package matrix

import (
	"fmt"
	"github.com/fxsjy/gonn/gonn"
	"testing"
)

func TestNNMatrix(t *testing.T) {
	CreateNN()
	// Загружем НС из файла.
	nn := gonn.LoadNN("nn_test")

	// Записываем значения в переменные:
	// hp - здоровье (0.1 - 1.0)
	// weapon - наличие оружия (0 - нет, 1 - есть)
	// enemyCount - количество врагов
	var hp float64 = 2.0
	var weapon float64 = 1.0
	var enemyCount float64 = 5.0

	// Получаем ответ от НС (массив весов)
	out := nn.Forward([]float64{hp, weapon, enemyCount})
	// Печатаем ответ на экран.
	fmt.Println(GetResult(out))
}
func GetResult(output []float64) string {
	max := -99999.0
	pos := -1
	// Ищем позицию нейрона с самым большим весом.
	for i, value := range output {
		if value > max {
			max = value
			pos = i
		}
	}

	// Теперь, в зависимости от позиции, возвращаем решение.
	switch pos {
	case 0:
		return "Атаковать"
	case 1:
		return "Красться"
	case 2:
		return "Убегать"
	case 3:
		return "Ничего не делать"
	}
	return ""
}

func CreateNN() {
	// Создаём НС с 3 входными нейронами (столько же входных параметров),
	// 16 скрытыми нейронами и
	// 4 выходными нейронами (столько же вариантов ответа)
	nn := gonn.DefaultNetwork(3, 16, 4, false)

	// Создаём массив входящих параметров:
	// 1 параметр - количество здоровья (0.1 - 1.0)
	// 2 параметр - наличие оружия (0 - нет, 1 - есть)
	// 3 параметр - количество врагов
	input := [][]float64{
		[]float64{0.5, 1, 1}, []float64{0.9, 1, 2}, []float64{0.8, 0, 1},
		[]float64{0.3, 1, 1}, []float64{0.6, 1, 2}, []float64{0.4, 0, 1},
		[]float64{0.9, 1, 7}, []float64{0.6, 1, 4}, []float64{0.1, 0, 1},
		[]float64{0.6, 1, 0}, []float64{1, 0, 0}}

	// Теперь создаём "цели" - те результаты, которые нужно получить
	target := [][]float64{
		[]float64{1, 0, 0, 0}, []float64{1, 0, 0, 0}, []float64{1, 0, 0, 0},
		[]float64{0, 1, 0, 0}, []float64{0, 1, 0, 0}, []float64{0, 1, 0, 0},
		[]float64{0, 0, 1, 0}, []float64{0, 0, 1, 0}, []float64{0, 0, 1, 0},
		[]float64{0, 0, 0, 1}, []float64{0, 0, 0, 1}}

	// Начинаем обучать нашу НС.
	// Количество итераций - 100000
	nn.Train(input, target, 100000)

	// Сохраняем готовую НС в файл.
	gonn.DumpNN("nn_test", nn)
}
