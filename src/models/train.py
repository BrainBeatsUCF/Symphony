from preprocess import generate_training_sequences, SEQUENCE_LENGTH, ROOT_PATH



OUTPUT_NEURONS = 45 # Equal to vocabulary size from the mapping.json
LOSS_FUNC = "sparse_categorical_crossentropy"
EPOCHS = 50 # 40 through 100 seems to work
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_NEURONS = [256] # number of neurons in the eternal layers
SAVE_MODEL_PATH = f"{ROOT_PATH}/model.h5"

def build_model(output_neurons, num_neurons, loss, learning_rate):

    # Create the model architecture, functionally!
    # If we say None for the shape, it lets us have any length input for the time steps
    # This is important because we might wanna feed it different length premade sequences
    input = keras.layers.Input(shape=(None, output_neurons))
    x = keras.layers.LSTM(num_neurons[0])(input) # This links the layers together
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(output_neurons, activation="softmax")(x)

    # compile the model
    model = keras.Model(input, output)
    model.compile(loss=loss,
                optimizer=keras.optimizers.Adam(lr=learning_rate),
                metrics=["accuracy"])

    model.summary()

    return model




def train(output_neurons, num_neurons, loss, learning_rate):

    # Generate the training sequences
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

    # build the network
    model = build_model(output_neurons, num_neurons, loss, learning_rate)

    # train the model
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # save the model
    model.save(SAVE_MODEL_PATH)


if __name__ == "__main__":
    train(OUTPUT_NEURONS, NUM_NEURONS, LOSS_FUNC, LEARNING_RATE)