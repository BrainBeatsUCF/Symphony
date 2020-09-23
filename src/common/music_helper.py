import os
import json
import music21 as m21
import tensorflow.keras as keras
import numpy as np
from typing import List

# durations are expressed in quarter length
ACCEPTABLE_DURATIONS = [
    0.25, # 16th note
    0.5, # 8th note
    0.75,
    1.0, # quarter note
    1.5,
    2, # half note
    3,
    4 # whole note
]

class MusicHelper:
    # TODO: Depdency Injection
    def __init__(self, file_helper, acceptable_durations=ACCEPTABLE_DURATIONS):
        """
        :param file_helper (FileHelper): This is a file_helper object to help with writing/reading files
        :param acceptable_durations (List[float]): This is the range of notes that are ok to add for the model
        :return MusicHelper: This is the constructor for the class
        """
        print("MusicHelper has been created...")
        self.acceptable_durations = acceptable_durations
        self._file_helper = file_helper

    def load_songs(self, dataset_path: str, file_type: str):
        """
        Loads all pieces of a specific file type

        :param dataset_path (str): Path to dataset
        :param file_stype (str): The file type you want to load
        :return songs (list of m21 streams): List containing all pieces
        """
        songs = []
        len_file_ext = len(file_type)

        # go through all the files in dataset and load them with music21
        for path, _, files in os.walk(dataset_path):
            for file in files:

                # consider only files of the target type
                if file[-len_file_ext:] == file_type:
                    song = m21.converter.parse(os.path.join(path, file))
                    songs.append(song)
        return songs

    def has_acceptable_durations(self, song) -> bool:
        """
        Boolean routine that returns True if piece has all acceptable duration, False otherwise.

        :param song (m21 stream):
        :return (bool):
        """
        for note in song.flat.notesAndRests:
            if note.duration.quarterLength not in self.acceptable_durations:
                return False
        return True

    def transpose_song(self, song, major_key: str, minor_key: str):
        """
        Transposes song to a specified major/minor key.
        If the song is in a major scale it is transposed to the specified key, vice versa

        Defaults are in place in case the programmer forgets them

        :param song (m21 stream): The song to transpose
        :param major_key (str): The musical major key you want to transpose to
        :param minor_key (str): The musical minor key you want to transpose to
        :return transposed_song (m21 stream):
        """

        # get key from the song
        parts = song.getElementsByClass(m21.stream.Part)
        measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
        key = measures_part0[0][4] # This is the hardcoded index of the key from m21 docs

        # estimate key using music21
        if not isinstance(key, m21.key.Key):
            key = song.analyze("key")

        # get interval for transposition. E.g., Bmaj -> Cmaj
        if key.mode == "major":
            interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
        elif key.mode == "minor":
            interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

        # transpose song by calculated interval
        tranposed_song = song.transpose(interval)
        return tranposed_song

    def encode_song(self, song, time_step=0.25):
        """
        Converts a score into a time-series-like music representation. Each item in the encoded list represents 'min_duration'
        quarter lengths. The symbols used at each step are: integers for MIDI notes, 'r' for representing a rest, and '_'
        for representing notes/rests that are carried over into a new time step. Here's a sample encoding:

            ["r", "_", "60", "_", "_", "_", "72" "_"]

        :param song (m21 stream): Piece to encode
        :param time_step (float): Duration of each time step in quarter length
        :return:
        """

        encoded_song = []

        for event in song.flat.notesAndRests:

            # handle notes
            if isinstance(event, m21.note.Note):
                symbol = event.pitch.midi # 60
            # handle rests
            elif isinstance(event, m21.note.Rest):
                symbol = "r"

            # convert the note/rest into time series notation
            steps = int(event.duration.quarterLength / time_step)
            for step in range(steps):

                # if it's the first time we see a note/rest, let's encode it. Otherwise, it means we're carrying the same
                # symbol in a new time step
                if step == 0:
                    encoded_song.append(symbol)
                else:
                    encoded_song.append("_")

        # cast encoded song to str
        encoded_song = " ".join(map(str, encoded_song))

        return encoded_song

    def convert_songs_to_int(self, songs: str, mapping_path:str):
        """ 
        Maps the symbol in the song to an integer as dictated by the mappings.json file you create
        earlier in training

        :param songs (str): The giant song string you are currently using
        :mapping_path (str): Path to your mappins.json file
        :return int_songs
        """
        int_songs = []
        
        # load mappings
        # TODO: Make the file_helper do this
        with open(mapping_path, "r") as fp:
            mappings = json.load(fp)

        # cast songs string to a list
        songs = songs.split()

        # map sings to int
        for symbol in songs:
            # TODO: Add some defensive coding strats here if key !exists 
            int_songs.append(mappings[symbol])

        return int_songs

    def create_mapping(self, songs: str, mapping_path: str) -> None:
        """
        Creates a json file that maps the symbols in the song dataset onto integers
        This mappins file is EXTREMELY important for further training, dont lose it!

        :param songs (str): String with all songs
        :param mapping_path (str): Path where to save mapping
        :return:
        """
        mappings = {}

        # identify the vocabulary
        songs = songs.split()
        vocabulary = list(set(songs))

        # create mappings
        for i, symbol in enumerate(vocabulary):
            mappings[symbol] = i

        # save voabulary to a json file
        # TODO: Have the file_helper do this
        with open(mapping_path, "w+") as fp:
            json.dump(mappings, fp, indent=4)

    def generate_training_sequences(self, sequence_length: int, single_file_dataset_path: str, mapping_path: str):
        """ 
        This creates representations of our data that can now be fed to an LSTM model

        :param sequence_length (int): The length of the "sliding window" we're using to refeed samples
        :param single_file_dataset_path (str): The file generated by create_single_file_dataset
        :param mapping_path (str): The mapping file that maps symbols to ints
        :return inputs, targets (3D Numpy Array): This is a 3d Numpy array/tensor for the model
        """
        # What this is trying to create in the model:
        # [11, 12, 13, 14, ...] -> inputs:[11, 12],  target:[13]

        # load songs and map them to int
        songs = self._file_helper.load_file_as_str(single_file_dataset_path)
        int_songs = self.convert_songs_to_int(songs, mapping_path)

        # generate the training sequences
        # 100 symbols, seq_len = 64, 100 - 64 = 36 sequences we can generate
        inputs = []
        targets = []
        num_sequences = len(int_songs) - sequence_length

        # This loops generates the training slices plus the target to predict
        for i in range(num_sequences):
            inputs.append(int_songs[i:i+sequence_length])
            targets.append(int_songs[i+sequence_length])

        # one-hot encode the sequences
        # inputs: (# of sequences, sequence length, vocabulary size)
        vocabulary_size = len(set(int_songs))
        inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size, dtype=np.uint8)
        targets = np.array(targets)

        # Inputs will be a 3D Numpy array as demonstrated above
        return inputs, targets


    def create_single_file_dataset(self, encoded_song_path: str, file_dataset_path: str, sequence_length: int) -> str:
        """
        Generates a file collating all the encoded songs and adding new piece delimiters. It then saves that to disc and returns the song

        :param dataset_path (str): Path to folder containing the encoded songs
        :param file_dataset_path (str): Path to file for saving songs in single txt file
        :param sequence_length (int): # of time steps to be considered for training
        :return songs (str): String containing all songs in dataset + delimiters
        """

        new_song_delimiter = "/ " * sequence_length
        songs = ""

        # load encoded songs and add delimiters
        for path, _, files in os.walk(encoded_song_path):
            for file in files:
                file_path = os.path.join(path, file)
                song = self._file_helper.load_file_as_str(file_path)
                songs = songs + song + " " + new_song_delimiter

        # remove empty space from last character of string
        songs = songs[:-1]

        # save string that contains all the dataset
        # TODO: Move to file helper + add defensive checks
        with open(file_dataset_path, "w+") as fp:
            fp.write(songs)

        return songs

    def preprocess_songs(self, dataset_path: str, song_txt_path: str, major_key: str, minor_key: str, file_type: str) -> None:
        """
        A method that encompasses many of the preprocessing operations needed for converting
        the songs to a helpful representation for our RNN/LSTM/Time Series centric models

        :param dataset_path (str): The complete path to the dataset you want to load
        :param song_text_path (str): The complete path where the song_txt file will be saved
        :param major_key (str): What major key we transpose songs to
        :param minor_key (str): What minor key we transpose songs to
        :return None: This method returns nothing
        """

        # load folk songs
        print("Loading songs...")
        songs = self.load_songs(dataset_path, file_type)
        print(f"Loaded {len(songs)} songs.")

        for i, song in enumerate(songs):

            # filter out songs that have non-acceptable durations
            if not self.has_acceptable_durations(song):
                continue

            # transpose songs to the major/minor key we want
            song = self.transpose_song(song, major_key, minor_key)

            # encode songs with music time series representation
            encoded_song = self.encode_song(song)

            # save songs to text file
            save_path = os.path.join(song_txt_path, str(i))
            # TODO: Have file helper do this + defensive checks
            with open(save_path, "w+") as fp:
                fp.write(encoded_song)

    # TODO: Ensure this is as genric as possible and applicate 
    def song_data_pipeline(self, pipeline_config: dict) -> None:
        """ 
        A single method that encapsulates the entire preprocessing pipeline for files. 

        :param pipeline_config (dict): A dict containing all the parameters and arguments for the methods being called.
        :return None: This method returns nothing
        """

        # TODO: Add better defensive coding, throughout the entire method really
        if (len(pipeline_config.keys()) < 6):
             raise Exception("Not enough keys, returning...")
        
        print("Entering song preprocessing...")
        self.preprocess_songs(
            pipeline_config['DATASET_PATH'], 
            pipeline_config['ENCODED_SONG_PATH'],
            pipeline_config['MAJOR_KEY'],
            pipeline_config['MINOR_KEY'],
            pipeline_config['FILE_TYPE']
        )
        
        songs = self.create_single_file_dataset(
            pipeline_config['ENCODED_SONG_PATH'],
            pipeline_config['SINGLE_FILE_DATASET_PATH'],
            pipeline_config['SEQUENCE_LENGTH']
        )

        self.create_mapping(songs, pipeline_config['MAPPING_PATH'])
        print("Finished preprocessing songs!")

        print("Generating training data...")
        inputs, targets = self.generate_training_sequences(
            pipeline_config['SEQUENCE_LENGTH'],
            pipeline_config['SINGLE_FILE_DATASET_PATH'],
            pipeline_config['MAPPING_PATH']
        )

        print("Finishing creating training data. Now returning....")
        return inputs, targets

