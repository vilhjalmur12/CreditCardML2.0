import logging
from services.Log.bcolors import bcolors
import datetime


class Log:

    def __init__(self, verbose=True, file_handler=False, file=None):
        self.file_handler = file_handler
        self.verbose = verbose

        if self.file_handler:
            if file == None:
                self.file = 'logs.log'
            else:
                self.file = file

        self.raw_msg = '{} - [{}] - Trainer - {} - {}'

    def warning(self, _class, message, exception=None):
        date = datetime.datetime.now()
        std_out = bcolors.WARNING + self.raw_msg.format('WARNING', date, _class, message)
        file_out = self.raw_msg.format('WARNING', date, _class, message) + '\n'

        if not exception == None:
            std_out = std_out + '\n' + str(exception) + '\n'
            file_out = file_out + str(exception) + '\n'

        if self.file_handler:
            with open(self.file, 'a+') as log_file:
                log_file.write(file_out)
        if self.verbose:
            std_out = std_out + bcolors.ENDC
            print(std_out)

    def info(self, _class, message):
        date = datetime.datetime.now()
        std_out = bcolors.OKGREEN + self.raw_msg.format('INFO', date, _class, message)
        file_out = self.raw_msg.format('INFO', date, _class, message) + '\n'

        if self.file_handler:
            with open(self.file, 'a+') as log_file:
                log_file.write(file_out)
        if self.verbose:
            std_out = std_out + bcolors.ENDC
            print(std_out)

    def debug(self, _class, message, exception=None):
        date = datetime.datetime.now()
        std_out = bcolors.OKBLUE + self.raw_msg.format('DEBUG', date, _class, message)
        file_out = self.raw_msg.format('DEBUG', date, _class, message) + '\n'

        if not exception == None:
            std_out = std_out + '\n' + str(exception) + '\n'
            file_out = file_out + str(exception) + '\n'

        if self.file_handler:
            with open(self.file, 'a+') as log_file:
                log_file.write(file_out)
        if self.verbose:
            std_out = std_out + bcolors.ENDC
            print(std_out)

    def error(self, _class, message, exception=None):
        date = datetime.datetime.now()
        std_out = bcolors.WARNING + self.raw_msg.format('ERROR', date, _class, message)
        file_out = self.raw_msg.format('ERROR', date, _class, message) + '\n'

        if not exception == None:
            std_out = std_out + '\n' + str(exception) + '\n'
            file_out = file_out + str(exception) + '\n'

        if self.file_handler:
            with open(self.file, 'a+') as log_file:
                log_file.write(file_out)
        if self.verbose:
            std_out = std_out + bcolors.ENDC
            print(std_out)
