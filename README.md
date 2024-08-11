# conti-mle
Homework for Senior Machine Learning Engineer - DevOps focus

This repository demonstrates the ML workflow and deployment of a Fashion MNIST image recognition model and service.
This is the recommended and chosen dataset as image processing is a relevant domain for the current position.

* The model is trained with PyTorch, PyTorch Lightning, and Torchvision libraries.
* The workflow is orchestrated with GitHub Actions.
  (GHA is preferred over Airflow for the orchestration because of the model's lightweight requirements)

The model training uses CrossEntropyLoss as a loss function, Adam (Adaptive Moment Estimation) as an optimiser, and Early stop to avoid overfitting the model.

LOCAL TEST
* execute the training script (train-torchlit-mlflow-faMNIST.py) in the training directory
* run inference.py under local-test for using the latest model artifact for infering output with a random input
* for local inference, you can also run server.py and send curl request to the Flask endpoint such as
* for having an input image you can run collect_mnist_image.py or just use sample_fashion_mnist.png
* please also set the local_run variable to True
- curl -X POST -F "file=@/Users/{your_name}/Projects/conti-mle/local-test/sample_fashion_mnist.png" http://localhost:5000/infer

* To test the Docker image locally, build and run the docker image:
- docker build --build-arg EXPERIMENT_ID=your_experiment_id --build-arg RUN_ID=your_run_id -t my-flask-app .
- e.g. docker build --build-arg EXPERIMENT_ID=737694674622074143 --build-arg RUN_ID=c09a656b6f624a72b6897ad6dcb7c122 -t my-flask-app . 
- docker run -p 5000:5000 -d my-flask-app
- you can also run docker_test_request.py in local-test directory
- similarly, you can run the same curl command to infer result