package main

import (
	"log"
	"neural_network/apiserver"
)

func main() {
	config := apiserver.CreateConfig()

	s := apiserver.New(config)
	if err := s.Start(); err != nil {
		log.Fatal(err)
	}
}
