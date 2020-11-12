# Symphony
Home of the generative models that power Brain Beats

![](https://github.com/BrainBeatsUCF/Symphony/workflows/PythonFormatting/badge.svg)


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

We use an Ubuntu base image for Docker. This is because of Fluidsynth, the virtual software synth for helping generate workable audio files. Although Fluidsynth does work on Alpine, Python is buggy on alpine images, and tensorflow cant compile because lack of a gcc. Using Ubuntu results in large images so be weary. 

Build the docker image
```
docker build -t docker-username/symphony .
```

Run the image
```
docker run -p 8000:8000 docker-username/symphony
```

To push the image to your docker repo use the command:
```
docker push docker-username/docker-reponame
```

## Debugging Docker & Dev tips
* Make sure you add the source of model data to your .dockerignore or you will have a very large docker image

