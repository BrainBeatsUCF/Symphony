# Symphony
Home of the generative models that power Brain Beats

## Local Instructions
* Clone the repo
* Navigate to the backend directory
* We reccommended you create a virtual environment. We use [pipenv](https://pypi.org/project/pipenv/)

Enter this in your command line:
```
pip install pipenv
pipenv shell
pipenv install
```
You can now run the project!. To run the API enter from the backend/ folder enter:
```
cd src/
uvicorn main:app --reload
```
or
```
uvicorn src.main:app --reload
```

If you run into any errors, double check those above commands

## Docker Instructions
* Install Docker to your machine
* Clone the repo
* Navigate to the backend/ directory

Build the docker image
```
docker build -t symphony-api .
```

Run the image
```
docker run -p 8000:8000 symphony-api
```

## Debugging Docker & Dev tips
* Make sure you add the source of model data to your .dockerignore or you will have a very large model

