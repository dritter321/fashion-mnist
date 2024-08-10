# conti-mle
Homework for Senior Machine Learning Engineer - DevOps focus

This repository demonstrates the ML workflow and deployment of an Fashion MNIST image recognition model and service.

* The model is trained with PyTorch Lightning and Torchvision libraries.
* The workflow is orchestrated with GitHub Actions.
  (GHA is preferred over Airflow for the orchestration because of the model's lightweight requirements)

The model training uses CrossEntropyLoss as a loss function, Adam (Adaptive Moment Estimation) as an optimiser, and Early stop to avoid overfitting the model.
