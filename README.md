Digit Recognition App
============
[![GitHub Stars](https://img.shields.io/github/stars/jordanhoare/digit-drawing-prediction.svg)](https://github.com/jordanhoare/digit-drawing-prediction/stargazers) [![GitHub Issues](https://img.shields.io/github/issues/jordanhoare/digit-drawing-prediction.svg)](https://github.com/jordanhoare/digit-drawing-prediction/issues) [![Current Version](https://img.shields.io/badge/version-0.5.0-green.svg)](https://github.com/jordanhoare/digit-drawing-prediction) 

![Alt Text](https://media.giphy.com/media/aIyeExuk11gpQTzk81/giphy.gif)

A python app that utilises a Lenet-5 Convolutional Neural Network architecture to identify user drawn digits (0 to 9). The model was trained on the MNIST dataset, using PyTorch, and spun up using FastAPI and a SQLite database. 


## Project Features
- [x] instantiate the project using pyenv, poetry
- [x] connect to SQLite with sqlalchemy object related mapping
- [x] add SemanticUI
- [x] build a JavaScript canvas that exports the image toDataURL as json string
- [x] configure CUDA for PyTorch modelling and train on the MNIST
- [x] dataURL to a greyscale 28x28, then transform array for input with model
- [ ] host via Heroku for a live demo capability


</br>

## Installation
To demo this application: clone the repo, navigate to the project folder and run `poetry shell` to open the venv.  Run the ` __main__.py ` file to start the server.  You can then access the web UI @ `http://127.0.0.1:8000/`


</br>

</br>


<p align="center">
    <a href="https://www.linkedin.com/in/jordan-hoare/">
        <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" />
    </a>&nbsp;&nbsp;
    <a href="https://www.kaggle.com/jordanhoare">
        <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white" />
    </a>&nbsp;&nbsp;
    <a href="mailto:jordanhoare0@gmail.com">
        <img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white" />
    </a>&nbsp;&nbsp;
</p>



