from midi2audio import FluidSynth
import os

for path, _, files in os.walk("./soundfonts"):
    for file in files:
        try:
            wav_file_name = f"./output/{file}.wav"
            fs = FluidSynth(f"./soundfonts/{file}")
            fs.midi_to_audio("./test_output.mid", wav_file_name)
        except Exception:
            print(f"Error processing file: {file} in {path}")