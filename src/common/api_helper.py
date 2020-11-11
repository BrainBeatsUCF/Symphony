import random

def delete_files(file):
    if file.endswith(".wav") or file.endswith(".mid"):
        return True
    else:
        return False

def determine_seed(mappings_path: str, file_helper):
    note_mappings = file_helper.loadJSON(mappings_path)
    note_hold = "_"
    rest = "r"
    song_end = "/"

    seed = []

    seed_range = random.randint(2, 6)
    note_mappings = list(note_mappings.keys())
    for i in range(seed_range):

        # make sure we dont generate a hold or rest at the beginning
        valid_note = False
        while not valid_note:
            new_note = random.choice(note_mappings)

            if new_note == rest and i >= 1:
                valid_note = True 

            if new_note != note_hold and new_note != song_end:
                valid_note = True
        
        seed.append(new_note)

        # generate a random duration
        note_duration = random.randint(1, 4)
        for _ in range(note_duration):
            seed.append(note_hold)

    return seed

def determine_metaparameters(emotion: str):
    if emotion == "happy":
        num_steps = random.randint(8, 32)
        max_seq_len = random.randint(num_steps, (num_steps*2))
        temperature = round(random.uniform(0.3, 1.5), 2)
    elif emotion == "calm":
        num_steps = random.randint(32, 64)
        max_seq_len = random.randint(num_steps, (num_steps*2))
        temperature = round(random.uniform(0.2, 0.8), 2)
    elif emotion == "melancholy":
        num_steps = random.randint(8, 42)
        max_seq_len = random.randint(num_steps, (num_steps*2))
        temperature = round(random.uniform(0.3, 1.0), 2)
    elif emotion == "surprised":
        num_steps = random.randint(8, 32)
        max_seq_len = random.randint(num_steps, (num_steps*2))
        temperature = round(random.uniform(0.8, 2.0), 2)
    else:
        print("No emotion found... setting random variables")
        num_steps = random.randint(8, 64)
        max_seq_len = random.randint(num_steps, (num_steps*2))
        temperature = round(random.uniform(0.3, 2.5), 2)

    return num_steps, max_seq_len, temperature