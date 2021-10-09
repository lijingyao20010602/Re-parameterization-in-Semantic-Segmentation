import json
import logging
import os

def setup_logger(name=None, output=None, for_file=False):
    # create a logger 
    logger = logging.getLogger(name) 
    logger.setLevel(logging.DEBUG) 

    # the handler format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') 

    # create a handler to output to terminal window
    if not for_file:
        ch = logging.StreamHandler() 
        ch.setLevel(logging.DEBUG) 
        ch.setFormatter(formatter) 
        logger.addHandler(ch) 

    # write log file
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")

        # create a handler to write log file
        fh = logging.FileHandler(filename) 
        fh.setLevel(logging.DEBUG) 
        fh.setFormatter(formatter) 
        logger.addHandler(fh)

    return logger


logging.basicConfig(level=logging.INFO, format='')
class Logger:
    def __init__(self):
        self.entries = {}

    def add_entry(self, entry):
        self.entries[len(self.entries) + 1] = entry

    def __str__(self):
        return json.dumps(self.entries, sort_keys=True, indent=4)