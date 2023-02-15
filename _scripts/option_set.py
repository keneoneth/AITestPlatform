# stores the custom options used during test run

from ailogger import ailogger

class OptionSet:

    def __init__(self, opt_train, opt_test, opt_model_path):

        self.opt_train = opt_train
        self.opt_test = opt_test
        self.opt_model_path = opt_model_path

        if self.opt_train == False and self.opt_test == True:
            try:
                assert self.opt_model_path is not None
            except:
                ailogger.exception("please specify model path for only testing i.e. train==False")
                raise

    def __str__(self) -> str:
        return "OptionSet params: opt_train:{} | opt_test:{} | opt_model_path:{}".format(self.opt_train, self.opt_test, self.opt_model_path)