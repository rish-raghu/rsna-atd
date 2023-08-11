import os

def makedir(dir):
    try:
        os.mkdir(dir)
    except OSError:
        pass
