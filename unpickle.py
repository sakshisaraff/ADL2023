import pickle

def unpickling(file_path):
    data = None  # Initialize data to None or a default value
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
    except FileNotFoundError:
        print("File not found.")
        return None  # Or handle the error as needed
    except pickle.UnpicklingError:
        print("Error in unpickling the file.")
        return None  # Or handle the error as needed
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None  # Or handle the error as needed

    return data




