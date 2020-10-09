FROM python:3.7-alpine

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN apk update \
    && apk add -u fluidsynth \
    tini \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8000
COPY . .
COPY ./src .
RUN ls

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]