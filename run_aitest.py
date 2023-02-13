#!/usr/bin/env python


"""
run_aitest.py/aitest - aitest driver command-line interface
"""

import os
import toml
import json
import logging
import argparse

# path to store the results of test run
PATH_TO_RESULT = "./results/"

# stores the compulsory keys needed in test run toml
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
        assert TomlKeys.KEY_TITLE in loaded_toml
        assert TomlKeys.KEY_DATASETS in loaded_toml
        assert TomlKeys.KEY_MODELS in loaded_toml
        assert TomlKeys.KEY_TESTCASES in loaded_toml
        assert TomlKeys.KEY_TESTRUN in loaded_toml

# stores the custom options used during test run
class OptionSet:

    def __init__(self, opt_train, opt_test, opt_model_path):

        self.opt_train = opt_train
        self.opt_test = opt_test
        self.opt_model_path = opt_model_path

        if self.opt_train == None and self.opt_test == None:
            self.opt_train = True
            self.opt_test = True

        if self.opt_train == None and self.opt_test == True:
            assert self.opt_model_path != None, "[error] please specify model path for testing if train==False"

    def __str__(self) -> str:
        return "opt_train:{}|opt_test:{}|opt_model_path:{}|opt_save_per_epoch:{}".format(self.opt_train, self.opt_test, self.opt_model_path, self.opt_save_per_epoch)

    @staticmethod
    def get_saveweight_cb(result_path, save_best_only=True, monitor='loss'):
        import tensorflow as tf
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=result_path,
            verbose=1,
            save_best_only=save_best_only,
            mode='auto',
            save_weights_only=True,
            monitor=monitor,
            period=1
        )

# test run class object
class TestRun:

    title = ""
    LOADED_DATASETS = {}
    LOADED_MODELS = {}
    LOADED_TESTCASES = {}

    @staticmethod
    def load_objects(datasets, models, testcases):
        TestRun.LOADED_DATASETS = datasets
        TestRun.LOADED_MODELS = models
        TestRun.LOADED_TESTCASES = testcases

    @staticmethod
    def set_title(title):
        TestRun.title = "".join(title.split())

    def __init__(self, testindex, dataset_key, model_key, testcase_key, out_format):
        self.testindex = testindex
        self.dataset_key = dataset_key
        self.model_key = model_key
        self.testcase_key = testcase_key
        self.out_format = out_format

    def __str__(self):
        return "dataset:{}|model:{}|testcase:{}".format(self.dataset_key, self.model_key, self.testcase_key)

    def out_json_name(self):
        return '%s/%s_%s.json' % (self.title, self.testcase_key, str(self.testindex))

    def run_test(self, opt_set):
        from _scripts.load_dataset import DataLoad
        data = DataLoad.load_dataset(
            self.dataset_key, TestRun.LOADED_DATASETS[self.dataset_key])
        from _scripts.load_model import ModelLoad
        model = ModelLoad.load_model(
            self.model_key, TestRun.LOADED_MODELS[self.model_key])
        from _scripts.load_testcase import TestcaseLoad
        testfunc = TestcaseLoad.load_testcase(
            self.testcase_key, TestRun.LOADED_TESTCASES[self.testcase_key])
        # run test function
        rets = testfunc(data=data, model_key=self.model_key, model=model, testfunc=testfunc,
                        testconfig=TestRun.LOADED_TESTCASES[self.testcase_key], result_path=os.path.join(PATH_TO_RESULT,self.title), opt_set=opt_set)
        
        assert isinstance(rets, list)
        for index, ret in enumerate(rets):
            
            if index >= len(self.out_format):
                logging.warning("results length does not match with out format length")
                continue

            if self.out_format[index] == "json":
                with open(os.path.join(PATH_TO_RESULT,self.out_json_name()), 'w') as f:
                    json.dump(ret, f, indent=2)
                    f.close()
            else:
                logging.warning("out format {} not supported yet".format(
                    self.out_format[index]))

# function to test run toml file
def run_toml(tomlfile, opt_set):

    with open(tomlfile) as f:

        loaded_toml = toml.load(f)
        TomlKeys.validate_toml(loaded_toml)
        logging.info("processing {} with title {}".format(
            f.name, loaded_toml[TomlKeys.KEY_TITLE]))
        # set testrun title
        TestRun.set_title(loaded_toml[TomlKeys.KEY_TITLE])
        # create a new folder under result path
        if not os.path.exists(os.path.join(PATH_TO_RESULT,TestRun.title)):
            os.mkdir(os.path.join(PATH_TO_RESULT,TestRun.title))
        # load datasets,models,testcases
        TestRun.load_objects(loaded_toml[TomlKeys.KEY_DATASETS],
                             loaded_toml[TomlKeys.KEY_MODELS], loaded_toml[TomlKeys.KEY_TESTCASES])
        # run test one by one
        for testindex, testrun_dict in enumerate(loaded_toml[TomlKeys.KEY_TESTRUN]):
            testrun = TestRun(testindex, testrun_dict[TomlKeys.KEY_DATASET], testrun_dict[TomlKeys.KEY_MODEL],
                              testrun_dict[TomlKeys.KEY_TESTCASE], testrun_dict[TomlKeys.KEY_OUTFORMAT])
            logging.info("running {}".format(testrun))
            testrun.run_test(opt_set)
        # test run completed
        logging.info("testrun {} with title {} done !!!".format(
            f.name, loaded_toml[TomlKeys.KEY_TITLE]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--toml", required=True,
                        help="toml files of testcase separated by ',' e.g. abc.toml,def.toml,ghl.toml")
    parser.add_argument("--train", required=False, default=None,
                        help="set the testcase to only train the model", action='store_true')
    parser.add_argument("--test", required=False, default=None,
                        help="set the testcase to only to test the specified model path", action='store_true')
    parser.add_argument("-m", "--model_path", required=False, default=None,
                        help="specify the model path for training/testing (compulsory for testing if train==False)")
    args = parser.parse_args()
    tomls_to_run = args.toml.split(",")

    # set up option set
    opt_set = OptionSet(args.train, args.test, args.model_path)

    # run toml files one by one
    for tomlfile in tomls_to_run:
        run_toml(tomlfile, opt_set)
