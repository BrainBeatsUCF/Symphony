import sys
sys.path.insert(0, "./common/")

import os
import uuid
from typing import Optional
from fastapi import Body, Request, FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from melody_generator import MelodyGenerator
from file_helper import FileHelper
from music_helper import MusicHelper
from fastapi.responses import FileResponse
from midi2audio import FluidSynth

# init our API, helpers, and constants
app = FastAPI()
file_helper = FileHelper()
music_helper = MusicHelper(file_helper)
model_paths = file_helper.loadJSON("./model_info.json")

# add our app middleware
origins = [
    "http://localhost:3000",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3232",
    "http://localhost:5000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# init our data models
class SampleRequest(BaseModel):
    instrument_name: str
    emotion: str
    seed: str
    num_steps: int
    max_seq_len: int
    temperature: float

@app.get("/ping")
def pong():
    """ Used to test if the API is alive """
    return {"Hello": "World"}

@app.post("/request")
async def testRequest(req: Request):
    """ You can use this to test what the body of the request looks like """
    print(await req.body())
    return {"Hello": "World"}


@app.get("/getSample")
async def getSample(sampleRequest: SampleRequest):
    if sampleRequest.instrument_name not in model_paths:
        raise HTTPException(status_code=404, detail="Instrument not found")

    # Delete any previous midi/wav files still present locally
    
    # Init our model
    melody_generator = MelodyGenerator(
        model_paths[sampleRequest.instrument_name]['model_path'],
        music_helper,
        file_helper,
        model_paths[sampleRequest.instrument_name]['mapping_path'],
        model_paths[sampleRequest.instrument_name]['sequence_length']
    )

    # Generate our melody from a seed
    melody = melody_generator.generate_melody(
        sampleRequest.seed,
        sampleRequest.num_steps,
        sampleRequest.max_seq_len,
        sampleRequest.temperature
    )

    # Save the melody to disk
    unique_id = str(uuid.uuid4())
    save_file_name = f"./output/{unique_id}.mid"
    melody_generator.save_melody(
        melody,
        save_file_name,
        format="midi"
    )

    # Convert the midi to wav
    wav_file_name = f"./output/{unique_id}.wav"
    fs = FluidSynth(model_paths[sampleRequest.instrument_name]['soundfont'])
    fs.midi_to_audio(save_file_name, wav_file_name)

    # Return the audio file
    return FileResponse(wav_file_name)

