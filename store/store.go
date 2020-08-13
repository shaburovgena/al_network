package store

import (
	"encoding/json"
	"github.com/fxsjy/gonn/gonn"
	"github.com/sirupsen/logrus"
	"neural_network/vector"
	"os"
)

const storePath = "store/result.json"

type Store struct {
	logger  *logrus.Logger
	NN      *gonn.NeuralNetwork
	Results map[string]int
	nnname  string
}

func New(nnname string) *Store {
	return &Store{
		logger:  logrus.New(),
		Results: initResults(),
		NN:      initNN(nnname),
		nnname:  nnname,
	}
}
func initNN(nnname string) *gonn.NeuralNetwork {
	nn := &gonn.NeuralNetwork{}
	if _, err := os.Stat(nnname); os.IsNotExist(err) {
		nn = gonn.DefaultNetwork(1600, 256, 26, false)
	} else {
		nn = gonn.LoadNN(nnname)
	}
	return nn
}
func initResults() map[string]int {
	file, err := os.Open(storePath)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	decoder := json.NewDecoder(file)
	result := map[string]int{}
	err = decoder.Decode(&result)
	if err != nil {
		panic(err)
	}
	return result
}

func (store *Store) getTargetObjectIndex(name string) int {
	for k, v := range store.Results {
		if k == name {
			return v
		}
	}
	return 0
}
func (store *Store) GetResult(incomeFile string) (string, error) {
	max := -99999.0
	pos := -1
	income, err := vector.GetVectorFromImage(incomeFile)
	if err != nil {
		return "", err
	}
	out := store.NN.Forward(income)
	// Ищем позицию нейрона с самым большим весом.

	for i, value := range out {
		if value > max {
			max = value
			pos = i
		}
	}
	store.logger.Info("Result number ", pos)
	for k, v := range store.Results {
		if v == pos {
			return k, nil
		}
	}
	return "Nothing", nil
}
func (store *Store) Train(files []string, trainObject string) error {
	trainVars := [][]float64{}
	targetObjectIndex := store.getTargetObjectIndex(trainObject)
	lenFiles := len(files)
	target := store.generateTarget(lenFiles, targetObjectIndex)
	for _, filename := range files {
		path := "train/files/" + filename
		x1, err := vector.GetVectorFromImage(path)
		if err != nil {
			return err
		}
		trainVars = append(trainVars, x1)
	}

	store.NN.Train(trainVars, target, 1000)
	gonn.DumpNN(store.nnname, store.NN)
	return nil
}
func (store *Store) ifNNExist() bool {
	if _, err := os.Stat(store.nnname); os.IsNotExist(err) {
		return false
	}
	return true
}

func (store *Store) generateTarget(count int, index int) [][]float64 {
	var target [][]float64

	for i := 0; i < count; i++ {
		var t []float64
		for j := 0; j < len(store.Results); j++ {
			if j == index {
				t = append(t, 1)
			} else {
				t = append(t, 0)
			}
		}
		target = append(target, t)
	}
	return target
}
