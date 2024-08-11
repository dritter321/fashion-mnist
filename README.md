# conti-mle
Homework for Senior Machine Learning Engineer - DevOps focus

This repository demonstrates the ML workflow and deployment of an Fashion MNIST image recognition model and service.

* The model is trained with PyTorch Lightning and Torchvision libraries.
* The workflow is orchestrated with GitHub Actions.
  (GHA is preferred over Airflow for the orchestration because of the model's lightweight requirements)

The model training uses CrossEntropyLoss as a loss function, Adam (Adaptive Moment Estimation) as an optimiser, and Early stop to avoid overfitting the model.

LOCAL TEST
* execute the training script (train-torchlit-mlflow-faMNIST.py) in the training directory
* run inference.py under local-test for using the latest model artifact for infering output with a random input
* for local inference, you can also run server.py and send curl request to the Flask endpoint such as
* for having an input image you can run collect_mnist_image.py or just use sample_fashion_mnist.png
curl -X POST -F "file=@/Users/{your_name}/Projects/conti-mle/local-test/sample_fashion_mnist.png" http://localhost:5000/infer
