package handler

import (
	"encoding/json"
	"github.com/gorilla/mux"
	"github.com/sirupsen/logrus"
	"io"
	"net/http"
	"neural_network/store"
	"os"
)

type Handler struct {
	logger *logrus.Logger
	store  *store.Store
}

func New(nnname string) *Handler {
	return &Handler{
		logger: logrus.New(),
		store:  store.New(nnname),
	}
}

func (handler *Handler) TrainNN(writer http.ResponseWriter, request *http.Request) {
	trainObject := mux.Vars(request)["trainObject"]
	request.ParseMultipartForm(32 << 20)
	files := []string{}
	for fileForm, _ := range request.MultipartForm.File {
		file, header, err := request.FormFile(fileForm)
		if err != nil {
			handler.logger.Error(err)
			handler.sendResponse(writer, http.StatusInternalServerError, err, nil)
		}
		filename := "train/files/" + header.Filename
		savedFile, err := os.Create(filename)
		_, err = io.Copy(savedFile, file)
		if err != nil {
			handler.logger.Error(err)
			handler.sendResponse(writer, http.StatusInternalServerError, err, nil)
		}
		files = append(files, header.Filename)

	}
	_, err := io.WriteString(writer, "Ok")
	if err != nil {
		handler.logger.Error(err)
		handler.sendResponse(writer, http.StatusInternalServerError, err, nil)
	}

	go func() {
		err := handler.store.Train(files, trainObject)
		if err != nil {
			handler.logger.Error(err)
			handler.sendResponse(writer, http.StatusInternalServerError, err, nil)
		}
	}()

}

func (handler *Handler) GetResult(writer http.ResponseWriter, request *http.Request) {
	file, header, err := request.FormFile("file")
	if err != nil {
		handler.logger.Error(err)
		handler.sendResponse(writer, http.StatusInternalServerError, err, nil)
	}
	filename := "train/income/" + header.Filename
	savedFile, err := os.Create(filename)
	defer savedFile.Close()

	_, err = io.Copy(savedFile, file)
	if err != nil {
		handler.logger.Error(err)
		handler.sendResponse(writer, http.StatusInternalServerError, err, nil)
	}
	result, err := handler.store.GetResult(filename)
	if err != nil {
		handler.logger.Error(err)
		handler.sendResponse(writer, http.StatusInternalServerError, err, nil)
	}
	_, err = io.WriteString(writer, result)
	if err != nil {
		handler.logger.Error(err)
		handler.sendResponse(writer, http.StatusInternalServerError, err, nil)
	}
}

func (handler *Handler) sendResponse(writer http.ResponseWriter, code int, err error, result interface{}) {
	js, _ := json.Marshal(result)
	if err != nil {
		http.Error(writer, err.Error(), code)
		handler.logger.Error(err)
		return
	}
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(code)
	_, err = writer.Write(js)
	if err != nil {
		handler.logger.Error(err)
		return
	}
}
