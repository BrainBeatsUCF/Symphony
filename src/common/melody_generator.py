import tensorflow.keras as keras
import json
import numpy as np
import music21 as m21
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH, ROOT_PATH


class MelodyGenerator:

    def __init__(self, model_path: str):
        self.model_path = model_path # we might not need this
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH # This acts as our song delimtter 
    
    def generate_melody(self, seed, num_steps, max_seq_len, temperature):
        """ 
        * seed - what we want to pass to the network and the network countinues that seed
        * num_steps - How far we want the network to predict
        * max_seq_len - How many steps of the seeds do we want to consider for the network (the seed will grow large), 
        so we pass the seed as a sliding window
        """

        # create seed with start symbols
        # it is passed to us as a str, let us turn it into a list
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # map seed to int
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):
            # limit the seed to max_seq_len
            seed = seed[-max_seq_len:] 


            # ont-hot encode the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))

            # Keras expects 3-dimensions for predict so we use numpy to add that
            # (1, max_seq_len, len(mappings))
            onehot_seed = onehot_seed[np.newaxis, ...]

            # inference stage, this will result in a distribution of probabilities, but we just want the first/most likely one
            # i.e: [0.]
            probabilities = self.model.predict(onehot_seed)[0]
            output_int = self._sample_with_temperature(probabilities, temperature)

            # update seed
            seed.append(output_int)

            # map int to our encoding
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            # check whether we're at the end of a melody
            if output_symbol == "/":
                break

            # update the melody
            melody.append(output_symbol)

        return melody


    def _sample_with_temperature(self, probabilities, temperature=0.7):
        """ 
        The temperature modifies the probability distribution
        the lower the temperature, the more deterministic the sampling is
        if the temperature is 1 that is just default settings
        higher temperature, more exploritive model
        """
        predicitions = np.log(probabilities) / temperature
        probabilities =  np.exp(predicitions) / np.sum(np.exp(predicitions))

        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities)

        return index

    def save_melody(self, melody, file_name, format="midi", step_duration=0.25,):
        """ 
            * The melody object you want to pass
            * the name of the saved file
            * the file format you want to save it in
            * The step duration you trained the model in, 0.25 is a sixteenth note
        """

        # create a music21 stream
        stream = m21.stream.Stream()

        # parse all the symbols in the melody and create note/rest objects
        # i.e: 60 _ _ _ r _ 62 _ ....
        start_symbol = None
        step_counter = 1

        for i, symbol in enumerate(melody):

            # handle case in which we have a note/rest
            # else handle case in which we have a prolongation sign "_"
            if symbol != "_" or (i + 1) == len(melody):
                # ensure we're dealing with notes/rests beyond the first symbol
                if start_symbol is not None:
                    quarter_length_duration = step_duration * step_counter # converts to the time scale of a note
    
                    # Handle rest
                    # Else Handle note
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)
                    
                    # reset to the other note
                    step_counter = 1
                start_symbol = symbol
            else:
                step_counter += 1

        # write the m21 stream to midi file
        stream.write(format, file_name)


def example():
    mg = MelodyGenerator(f"{ROOT_PATH}/model2.h5")
    seed = "60 _ _ _ 60 _ 55 _ 57 _ 55 _ 60 _"
    melody = mg.generate_melody(seed, 250, SEQUENCE_LENGTH, 1)
    mg.save_melody(melody, f"{ROOT_PATH}/test.midi")
    print(melody)

if __name__ == "__main__":
    example()








