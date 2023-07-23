import os
import time
import threading
import sched

# scheduler object to clean that saved_img folder Amy made every 3min so we don't run out of space
scheduler = sched.scheduler(time.time, time.sleep)

def delayed_delete(path, delay):
    def delete_files():
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
                
    # schedule the delete_files function to run after `delay` seconds
    scheduler.enter(delay, 1, delete_files)
    
    # start a new thread that will run the scheduler
    threading.Thread(target=scheduler.run).start()

# delete all files in the `saved_img` directory after 3 minutes
delayed_delete('../AThousandWords/objects/saved_img', 3*60)
