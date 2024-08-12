# conti-mle
Homework for Senior Machine Learning Engineer - DevOps focus

## Description

This repository demonstrates the ML workflow and deployment of a Fashion MNIST image recognition model and service.
This is the recommended and chosen dataset as image processing is a relevant domain for the current position.

* The model is trained with PyTorch, PyTorch Lightning, and Torchvision libraries.
* The workflow is orchestrated with GitHub Actions.
  (GHA is preferred over Airflow for the orchestration because of the model's lightweight requirements)

The model training uses CrossEntropyLoss as a loss function, Adam (Adaptive Moment Estimation) as an optimiser, and Early stop to avoid overfitting the model.

## Local Testing

* before running the training script, execute `/preprocessing/preprocessing.py` to make the data available
* execute the training script `/training/train-torchlit-mlflow-faMNIST.py` in the training directory
* run `/local-test/inference.py`for using the latest model artifact for infering output with a random input
* for local inference, you can also run `/server/server.py` and send curl request to the Flask endpoint such as
* for having an input image you can run collect_mnist_image.py or just use `/local-test/sample_fashion_mnist.png`
* please also set the local_run variable to True, in case you run Flash server locally
```
curl -X POST -F "file=@/Users/{your_name}/Projects/conti-mle/local-test/sample_fashion_mnist.png" http://localhost:5000/infer
```

* To test the Docker image locally, build and run the docker image (the second one is contains the model run included):
```
docker build --build-arg EXPERIMENT_ID=your_experiment_id --build-arg RUN_ID=your_run_id -t my-flask-app .
```
```
docker build --build-arg EXPERIMENT_ID=737694674622074143 --build-arg RUN_ID=c09a656b6f624a72b6897ad6dcb7c122 -t my-flask-app . 
```
```
docker run -p 5000:5000 -d my-flask-app
```
- you can also run `/local-test/docker_test_request.py` in local-test directory
- similarly, you can run the same curl command to infer result

* A GHA workflow is also defined to publish the image Docker Hub, so you might also pull the latest published image (or even publish a new one)
```
docker pull dritter3/fashion-mnist-flask-app:latest
```  
* Some unit testing is added to GHA, as well.

## Theoretical Questions

### Testing

* Unit testing is included in GHA, utility functions can be unit tested and included before any builds.
* Functional tests can be also included.
* Data Loading and its performance needs to be measured and optimized further.
* Pipeline needs to be tested for its effectiveness and usability
* in `local-test`, inferental script and request script is provided to test functionality during development
* As much test should be included in automatic GHA for functional or code quality checks.

### Optimization, Bottlenecks, First Steps

* Data Loading and preprocessing requires optimal batch size. Experimenting is needed. For this local testing, it wasn't necessary here but parallel loading of DataLoader with appropriate number of num_workers needs to be found.
* For model training, using Early Stop, and storing Model Checkpoints might be unnecessary, and might require too much unncessary processing.
* Using mixed precision is only beneficial if it's appropriately chosen for hardware.
* Setting up monitoring tools, might help further optimization.
* In case of larger jobs, finding an optimal amount of logging might be necessary, too.
* Maintaining and testing up-to-date libraries

### multi-GPU considerations

* Parallelization techniques need to be considered for performance.
* PyTorch Training configuration can be optimized for multi-GPU processing ('gpu', 'ddp').
* Model can be parallelized, either in multi-GPU setup.
* Data parallelism is preferred of Model parallelism in case of limited resources.
- easier setup, scalability, better separation of processing

### Elective Tasks

* PyTorch Lightning implemented
* MLFlow is used as a tracking tool
* Two GHA workflows are included
