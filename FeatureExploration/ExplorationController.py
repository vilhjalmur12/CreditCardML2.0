import pandas as pd


class ExplorationController:
    def __init__(self, environ='local'):
        if environ == 'local':
            self.local = True
        else:
            self.local = False
