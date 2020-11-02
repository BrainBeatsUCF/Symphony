import tensorflow.keras as keras
import tensorflow as tf
import json
import numpy as np
import music21 as m21
from typing import List
from file_helper import FileHelper
from music_helper import MusicHelper


class MelodyGenerator:
    def __init__(self, model_path: str, file_helper: FileHelper, mapping_path: str, sequence_length: int, cpu = True):
        """
        Our constructor for the MelodyGenerator class

        :param model_path (str): Where the .h5 model file is being stored
        :param file_helper (FileHelper):
        :param mapping_path (str): Where are the mappings for the model being stored?
        :param sequence_length(int):
        """

        # The CPU is normally fast enough to do inference, and this makes local development easier
        if cpu:
            with tf.device('/cpu:0'):
                self.model = keras.models.load_model(model_path)
        else:
            self.model = keras.models.load_model(model_path)

        self._file_helper = file_helper
        self._start_symbols = ["/"] * sequence_length # This acts as our song delimtter 
        self._mappings = self._file_helper.loadJSON(mapping_path)

    
    def generate_melody(self, seed: str, num_steps: int, max_seq_len: int, temperature: float) -> List[str]:
        """ 
        Generates a melody from a starting seed and returns a list of symbols that can undergo
        additional processing

        :param seed (str): what we want to pass to the network and the network countinues that seed
        :param num_steps (int): How far we want the network to predict
        :param max_seq_len (int): ow many steps of the seeds do we want to consider for the network (the seed will grow large)
        :param temperature (float): How creative we want our model to be
        :return (List[str]):
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

            # one-hot encode the seed
            # The mappings is the "vocabulary" of the object
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


    def _sample_with_temperature(self, probabilities: List[float], temperature: float) -> int:
        """ 
        Uses some clever math to pick an encoding based on model output probabilities
        
        :param probabilities (List[float]): The output of the model associated with probabilites
        :param temperature (float): How creative we want our model to be
        :return (int):
        """

        predicitions = np.log(probabilities) / temperature
        probabilities =  np.exp(predicitions) / np.sum(np.exp(predicitions))

        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities)

        return index

    def save_melody(self, melody: List[str], file_name: str, format="mid", step_duration=0.25):
        """ 
        Converts our list to a m21 stream and saves the melody to the desired file format

        :param melody (List[str]): Our melody list to be saved on disk
        :param file_name (str): The name of the file we want to save including model path
        :param format (str): The music file format you want to save to, music21 supports very little so this should always be midi
        :param step_duration (float): Music theory stuff, the smallest note used, make sure this matches what you trained the model with
        :return (None): This function returns nothing, it saves to disk
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
        try:
            stream.write(format, file_name)
        except Exception as e:
            print(f"Exception when trying to write file: {file_name} with error: {e}")