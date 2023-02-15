# path to store the results of test run

import os
from ailogger import ailogger

class PathConfig:

    TXT_AITESTPLATFORM = 'AITESTPLATFORM'

    @staticmethod
    def get_results_path(extra=''):
        return os.path.join(os.environ[PathConfig.TXT_AITESTPLATFORM],'results',extra)

    @staticmethod
    def get_datasets_path(extra=''):
        return os.path.join(os.environ[PathConfig.TXT_AITESTPLATFORM],'datasets',extra)

    @staticmethod
    def get_models_path(extra=''):
        return os.path.join(os.environ[PathConfig.TXT_AITESTPLATFORM],'models',extra)

    @staticmethod
    def get_testcases_path(extra=''):
        return os.path.join(os.environ[PathConfig.TXT_AITESTPLATFORM],'testcases',extra)

    @staticmethod
    def validate():
        try:
            assert os.environ[PathConfig.TXT_AITESTPLATFORM]
        except:
            ailogger.exception('env var AITESTPLATFORM is not set')