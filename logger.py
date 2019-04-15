import logging
import os
import config

class Logger:
    def __init__(self, file):
        """
        Initialize Logger.
        """

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        # Info logging save into file
        resulthandler = logging.FileHandler(file)
        resulthandler.setLevel(logging.INFO)
        filefmt = logging.Formatter('%(asctime)s: %(message)s')
        resulthandler.setFormatter(filefmt)
        
        # Debug logging only print 
        debuger = logging.StreamHandler()
        debuger.setLevel(logging.DEBUG)
        filefmt = logging.Formatter('%(filename)s-%(funcName)s-%(lineno)d: %(message)s')
        debuger.setFormatter(filefmt)
        debugerfilter = logging.Filter()
        debugerfilter.filter = lambda record: record.levelno < logging.WARNING
        debuger.addFilter(debugerfilter)

        # Add Handlers
        self.logger.addHandler(resulthandler)
        self.logger.addHandler(debuger)


Log = Logger(config.debug_log_file)
