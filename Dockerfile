FROM python:3.10-slim-buster
WORKDIR /usr/src/app

COPY ./server/server.py /usr/src/app/server/server.py
COPY ./training/__init__.py /usr/src/app/training/__init__.py
COPY ./training/model.py /usr/src/app/training/model.py
COPY ./requirements.txt /usr/src/app
RUN pip install --no-cache-dir -r requirements.txt

ARG EXPERIMENT_ID
ARG RUN_ID
COPY ./training/mlruns/${EXPERIMENT_ID}/${RUN_ID}/model_artifact/model.pth /usr/src/app/model.pth

EXPOSE 5000
CMD ["python", "server/server.py"]