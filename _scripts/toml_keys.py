# stores the compulsory keys needed in test run toml
from ailogger import ailogger

class TomlKeys:

    KEY_TITLE = "title"
    KEY_DATASETS = "datasets"
    KEY_MODELS = "models"
    KEY_TESTCASES = "testcases"
    KEY_TESTRUN = "testrun"

    KEY_DATASET = "dataset"
    KEY_MODEL = "model"
    KEY_TESTCASE = "testcase"
    KEY_OUTFORMAT = "out_format"

    def validate_toml(loaded_toml):
        try:
            assert TomlKeys.KEY_TITLE in loaded_toml
            assert TomlKeys.KEY_DATASETS in loaded_toml
            assert TomlKeys.KEY_MODELS in loaded_toml
            assert TomlKeys.KEY_TESTCASES in loaded_toml
            assert TomlKeys.KEY_TESTRUN in loaded_toml
        except:
            ailogger.exception("toml validation failed")
            raise