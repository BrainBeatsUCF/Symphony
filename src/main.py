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
from api_helper import determine_seed, delete_files, determine_metaparameters
from music_helper import MusicHelper
from fastapi.responses import FileResponse
from midi2audio import FluidSynth
from decouple import config

# init our API, helpers, and constants
app = FastAPI()
file_helper = FileHelper()
music_helper = MusicHelper(file_helper)
model_paths = file_helper.loadJSON("./model_info.json")
debug_mode = int(config('SYMPHONY_DEBUG'))

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

@app.get("/ping")
def pong():
    """ Used to test if the API is alive """
    return {"Hello": "World"}

@app.post("/request")
async def testRequest(req: Request):
    """ You can use this to test what the body of the request looks like """
    print(await req.body())
    return {"Hello": "World"}


@app.post("/getSample")
async def getSample(sampleRequest: SampleRequest):
    if sampleRequest.instrument_name not in model_paths:
        raise HTTPException(status_code=404, detail="Instrument not found")

    # Delete any previous midi/wav files still present locally
    if debug_mode == 0:
        try:
            directory = "./output"
            files_in_directory = os.listdir(directory)
            files_to_delete = [file for file in files_in_directory if delete_files(file)]
            for file in files_to_delete:
                path_to_file = os.path.join(directory, file)
                os.remove(path_to_file)
        except Exception as e:
            print(f"Something went wrong with deleting thge old files: {e}")
    else:
        print("In debug mode, not deleting old output")
    
    # Init our model
    melody_generator = MelodyGenerator(
        model_paths[sampleRequest.instrument_name]['model_path'],
        file_helper,
        model_paths[sampleRequest.instrument_name]['mapping_path'],
        model_paths[sampleRequest.instrument_name]['sequence_length']
    )

    seed = determine_seed(model_paths[sampleRequest.instrument_name]['mapping_path'], file_helper)
    num_steps, max_seq_len, temperature = determine_metaparameters(sampleRequest.emotion)

    if debug_mode != 0:
        print(f"The seed is: {seed}")

    # Generate our melody from a seed
    melody = melody_generator.generate_melody(
        seed,
        num_steps,
        max_seq_len,
        temperature
    )

    # Save the melody to disk
    unique_id = str(uuid.uuid4())
    save_file_name = f"./output/{unique_id}.mid"
    melody_generator.save_melody(
        melody,
        save_file_name,
        sampleRequest.emotion,
        format="midi"
    )

    # Convert the midi to wav
    wav_file_name = f"./output/{unique_id}.wav"
    fs = FluidSynth(model_paths[sampleRequest.instrument_name]['soundfont'])
    fs.midi_to_audio(save_file_name, wav_file_name)

    # Return the audio file
    return FileResponse(wav_file_name)

