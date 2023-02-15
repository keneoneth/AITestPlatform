import importlib

# data load class
class DataLoad:
    @staticmethod
    def load_dataset(dataset_key,d):
        dataset_module = importlib.import_module("datasets."+dataset_key+"."+d["loadfname"])
        return getattr(dataset_module, d["loadobj"])