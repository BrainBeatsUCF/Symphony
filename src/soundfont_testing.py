import sys
sys.path.insert(0, "./common/")

from api_helper import determine_seed, delete_files, determine_metaparameters
from melody_generator import MelodyGenerator
from file_helper import FileHelper
from music_helper import MusicHelper
from midi2audio import FluidSynth
import random
import os

file_helper = FileHelper()
music_helper = MusicHelper(file_helper)
model_paths = file_helper.loadJSON("./model_info.json")
models_to_try = ["JazzDrums", "FunkDrums", "Guitar", "Drums"]
emotions = ["happy","melancholy","surprised","calm"]

def generate_melodies():
    for model in models_to_try:
        seed = determine_seed(model_paths[model]['mapping_path'], file_helper)
        num_steps, max_seq_len, temperature = determine_metaparameters(random.choice(emotions))

        melody_generator = MelodyGenerator(
        model_paths[model]['model_path'],
        file_helper,
        model_paths[model]['mapping_path'],
        64
        )

        melody = melody_generator.generate_melody(
        seed,
        num_steps,
        max_seq_len,
        temperature
        )

        save_file_name = f"./output/{model}_test_output.mid"
        melody_generator.save_melody(
            melody,
            save_file_name,
            random.choice(emotions),
            format="midi"
        )

        try_soundfonts(model)



def try_soundfonts(model_name):
    for path, _, files in os.walk("./soundfonts"):
        for file in files:
            try:
                wav_file_name = f"./output/{model_name}/{model_name}_{file}.wav"
                fs = FluidSynth(f"./soundfonts/{file}")
                fs.midi_to_audio(f"./output/{model_name}_test_output.mid", wav_file_name)
            except Exception:
                print(f"Error processing file: {file} in {path}")


generate_melodies()