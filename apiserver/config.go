package apiserver

import (
	"encoding/json"
	"errors"
	"flag"
	"os"
)

const configPath = "configs/config.json"

type Config struct {
	BindPort string
	LogLevel string
	NNName string
}

func newConfig() *Config {
	return &Config{
		BindPort: getParam("BIND_PORT", ":8080"),
		LogLevel: getParam("LOG_LEVEL", "debug"),
		NNName: getParam("NN_NAME", "train/nn/nn_test"),
	}
}
func CreateConfig() *Config {
	flag.Parse()

	config := newConfig()

	return config
}

func getParam(key string, defaultVal string) string {
	value, exists := os.LookupEnv(key)
	if exists {
		return value
	}
	file, err := os.Open(configPath)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	decoder := json.NewDecoder(file)
	config := map[string]string{}
	err = decoder.Decode(&config)
	if err != nil {
		panic(err)
	}
	valueFromConf, exist := config[key]
	if !exist {
		if defaultVal == "" {
			panic(errors.New("Missing required parameter " + key))
		} else {
			return defaultVal
		}
	}
	return valueFromConf
}
