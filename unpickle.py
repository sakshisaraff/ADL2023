import pickle

def unpickling(file_path)

    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
    except FileNotFoundError:
        print("File not found.")
    except pickle.UnpicklingError:
        print("Error in unpickling the file.")

    return data

