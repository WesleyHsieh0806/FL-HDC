import os
import shutil
# delete all files in directories of clients
for dir in os.listdir():
    if ("client" in dir) and (os.path.isdir(dir)):
        shutil.rmtree(dir)
os.remove("Setup.pickle")
os.remove("Global_model/global_model_dict.pickle")
