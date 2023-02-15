import importlib


# model load class
class ModelLoad:
    @staticmethod
    def load_model(model_key,model_dict):
        model_module = importlib.import_module("models."+model_key)
        return getattr(model_module, model_dict["modelname"])