from midi2audio import FluidSynth

fs = FluidSynth('./converter/soundfonts/General_MIDI_64_1.6.sf2')
fs.midi_to_audio('./output/test.mid', './output/output.wav')