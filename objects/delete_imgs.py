import os
import threading

# Function to delete files at a specified path
def delete_files(path):
    # Iterate over each file in the specified directory
    for filename in os.listdir(path):
        # Generate the full path for the file
        file_path = os.path.join(path, filename)
        try:
            # If the file_path is a file or a symbolic link, then delete it
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            # If the file_path is a directory, then remove it
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        # If there's any issue during file deletion, catch the exception and print it
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

# Function to initiate delayed deletion of files
def delayed_delete(path, delay):
    # Delete the files at the path
    delete_files(path)
    # Schedule the next call for delayed_delete function after `delay` seconds.
    # The function calls itself, creating a cycle that continues every `delay` seconds.
    threading.Timer(delay, delayed_delete, [path, delay]).start()




