import sys
sys.path.insert(0, "./../../common/")

from file_helper import FileHelper
from music_helper import MusicHelper
from pathlib import Path
import tensorflow.keras as keras


file_helper = FileHelper()
music_helper = MusicHelper(file_helper)

mappings = file_helper.loadJSON("./song_mappings.json")
OUTPUT_NEURONS = len(mappings.keys())  # Equal to vocabulary size from the mapping.json
LOSS_FUNC = "sparse_categorical_crossentropy"
EPOCHS = 50
LEARNING_RATE = 0.001
BATCH_SIZE = 16
NUM_NEURONS = [256] # number of neurons in the eternal layers
ROOT_PATH = Path.cwd()
SAVE_MODEL_PATH = f"{ROOT_PATH}/FolkLSTM.h5"

# Used with the music helper class
pipeline_config = {
    'DATASET_PATH': f"{ROOT_PATH}/data",
    'ENCODED_SONG_PATH': f"{ROOT_PATH}/processed_songs/",
    'MAJOR_KEY': "C",
    'MINOR_KEY': "A",
    'SINGLE_FILE_DATASET_PATH': f"{ROOT_PATH}/massive_song_file_data.txt",
    'MAPPING_PATH': f"{ROOT_PATH}/song_mappings.json",
    'SEQUENCE_LENGTH': 64,
    'FILE_TYPE': "krn",
    'STAGE': 0
}

def build_model(output_neurons, num_neurons, loss, learning_rate):
    # Create the model architecture, functionally!
    # If we say None for the shape, it lets us have any length input for the time steps
    # This is important because we might wanna feed it different length premade sequences
    input = keras.layers.Input(shape=(None, output_neurons))
    x = keras.layers.LSTM(num_neurons[0])(input) # This links the layers together
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(output_neurons, activation="softmax")(x),

    # compile the model
    model = keras.Model(input, output)
    model.compile(loss=loss,
                optimizer=keras.optimizers.Adam(lr=learning_rate),
                metrics=["accuracy"])
    model.summary()
    return model

# Generate the training sequences
inputs, targets = music_helper.song_data_pipeline(pipeline_config)

# build the network
model = build_model(OUTPUT_NEURONS, NUM_NEURONS, LOSS_FUNC, LEARNING_RATE)

# train the model
model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

# save the model
model.save(SAVE_MODEL_PATH)

