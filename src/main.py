import sys
sys.path.insert(0, "./common/")

from typing import Optional
from fastapi import Body, Request, FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from melody_generator import MelodyGenerator
from file_helper import FileHelper
from music_helper import MusicHelper
from fastapi.responses import StreamingResponse



# init our API + any constants
app = FastAPI()

model_paths = {
    'FolkLSTM': {
        'model_path': './models/FolkLSTM/FolkLSTM-Draft1.h5',
        'mapping_path': './models/FolkLSTM/song_mappings.json',
        'sequence_length': 64}
}

file_helper = FileHelper()
music_helper = MusicHelper(file_helper)

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
    
    melody_generator = MelodyGenerator(
        model_paths[sampleRequest.instrument_name]['model_path'],
        music_helper,
        file_helper,
        model_paths[sampleRequest.instrument_name]['mapping_path'],
        model_paths[sampleRequest.instrument_name]['sequence_length']
    )

    melody = melody_generator.generate_melody(
        sampleRequest.seed,
        sampleRequest.num_steps,
        sampleRequest.max_seq_len,
        sampleRequest.temperature
    )

    melody_generator.save_melody(
        melody,
        'test.midi',
        format="midi"
    )

    try:
        sample_audio = file_helper.readBytes("./test.midi")
    except Exception:
        # TODO: Use correct HTTP verb
        raise HTTPException(status_code=500, detail="Unable to read bytes for file")

    return StreamingResponse(sample_audio, media_type="audio/midi")

