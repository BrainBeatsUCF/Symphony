FROM ubuntu:20.04

RUN apt update -y
RUN apt upgrade -y
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN apt install python3-pip -y
RUN  apt install fluidsynth -y

COPY requirements.txt .
RUN pip3 install -r requirements.txt



EXPOSE 8000
COPY . .
COPY ./src .
RUN ls

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]