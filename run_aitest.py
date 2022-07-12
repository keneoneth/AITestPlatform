from ast import arg
import os
import toml
import argparse
import json


PATH_TO_TESTRUN = "./testrun/"
PATH_TO_TESTCASE = "./testcases/"
PATH_TO_MODEL = "./models/"
PATH_TO_RESULT = "./results/"
PATH_TO_DATASET = "./datasets/"

KEY_TITLE = "title"
KEY_DATASETS = "datasets"
KEY_MODELS = "models"
KEY_TESTCASES = "testcases"
KEY_TESTRUN = "testrun"

class OptionSet:

    def __init__(self,opt_train,opt_test,opt_model_path):

        self.opt_train = opt_train
        self.opt_test = opt_test
        self.opt_model_path = opt_model_path

        if self.opt_train == None and self.opt_test == None:
            self.opt_train = True
            self.opt_test = True

        if self.opt_train == None and self.opt_test == True:
            assert self.opt_model_path != None, "[error] please specify model path for testing if train==False"

    def __str__(self) -> str:
        return "opt_train:{}|opt_test:{}|opt_model_path:{}|opt_save_per_epoch:{}".format(self.opt_train,self.opt_test,self.opt_model_path,self.opt_save_per_epoch)

    @staticmethod
    def get_saveweight_cb(result_path,save_best_only=True,monitor='loss'):
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

class TestRun:

    KEY_DATASET = "dataset"
    KEY_MODEL = "model"
    KEY_TESTCASE = "testcase"
    KEY_OUTFORMAT = "out_format"

    title = ""
    LOADED_DATASETS = {}
    LOADED_MODELS = {}
    LOADED_TESTCASES = {}
    
    @staticmethod
    def load_objects(datasets,models,testcases):
        TestRun.LOADED_DATASETS = datasets
        TestRun.LOADED_MODELS = models
        TestRun.LOADED_TESTCASES = testcases
    
    @staticmethod
    def set_title(title):
        TestRun.title = "".join(title.split())

    def __init__(self,testindex,dataset_key,model_key,testcase_key,out_format):
        self.testindex = testindex
        self.dataset_key = dataset_key
        self.model_key = model_key
        self.testcase_key = testcase_key
        self.out_format = out_format
    
    def __str__(self):
        return "dataset:{}|model:{}|testcase:{}".format(self.dataset_key,self.model_key,self.testcase_key)
    
    def out_json_name(self):
        return '%s/%s_%s.json'%(self.title,self.testcase_key,str(self.testindex))

    def run_test(self,opt_set):
        from _scripts.load_dataset import DataLoad
        data = DataLoad.load_dataset(self.dataset_key,TestRun.LOADED_DATASETS[self.dataset_key])
        from _scripts.load_model import ModelLoad
        model = ModelLoad.load_model(self.model_key,TestRun.LOADED_MODELS[self.model_key])
        from _scripts.load_testcase import TestcaseLoad
        testfunc = TestcaseLoad.load_testcase(self.testcase_key,TestRun.LOADED_TESTCASES[self.testcase_key])
        rets = testfunc(data=data,model_key=self.model_key,model=model,testfunc=testfunc,testconfig=TestRun.LOADED_TESTCASES[self.testcase_key],result_path=PATH_TO_RESULT+'/'+self.title+"/",opt_set=opt_set)
        assert isinstance(rets,list)
        for index,ret in enumerate(rets):
            if not index < len(self.out_format):
                print("[warning] results is more than out format length")
                continue
            if self.out_format[index] == "json":
                with open(PATH_TO_RESULT+self.out_json_name(), 'w') as f:
                    json.dump(ret, f)
                    f.close()
            else:
                print("[warning] out format {} not supported yet".format(self.out_format[index]))
        
def validate_toml(loaded_toml):
    assert KEY_TITLE in loaded_toml
    assert KEY_DATASETS in loaded_toml
    assert KEY_MODELS in loaded_toml
    assert KEY_TESTCASES in loaded_toml
    assert KEY_TESTRUN in loaded_toml

def run_toml(tomlfile,opt_set):
    # print(opt_train,opt_test,opt_model_path,opt_save_per_epoch)
    with open(tomlfile) as f:
        loaded_toml = toml.load(f)
        validate_toml(loaded_toml)
        print("[info] processing {} with title {}".format(f.name,loaded_toml[KEY_TITLE]))
        TestRun.set_title(loaded_toml[KEY_TITLE])
        if not os.path.exists(PATH_TO_RESULT+TestRun.title):
            os.mkdir(PATH_TO_RESULT+TestRun.title)
        TestRun.load_objects(loaded_toml[KEY_DATASETS],loaded_toml[KEY_MODELS],loaded_toml[KEY_TESTCASES])
        for testindex,testrun_dict in enumerate(loaded_toml[KEY_TESTRUN]):
            testrun = TestRun(testindex,testrun_dict[TestRun.KEY_DATASET],testrun_dict[TestRun.KEY_MODEL],testrun_dict[TestRun.KEY_TESTCASE],testrun_dict[TestRun.KEY_OUTFORMAT])
            print("[info] running {}".format(testrun))
            testrun.run_test(opt_set)
        print("[info] testrun {} with title {} done !!!".format(f.name,loaded_toml[KEY_TITLE]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--toml",required=True,help="toml files of testcase e.g. abc.toml,def.toml,ghl.toml")
    parser.add_argument("--train",required=False,default=None,help="set the testcase to only train the model",action='store_true')
    parser.add_argument("--test",required=False,default=None,help="set the testcase to only to test the specified model path",action='store_true')
    parser.add_argument("-m","--model_path",required=False,default=None,help="specify the model path for training/testing (compulsory for testing if train==False)")
    args = parser.parse_args()
    tomls_to_run = args.toml.split(",")
    
    opt_set = OptionSet(args.train,args.test,args.model_path)

    for tomlfile in tomls_to_run:
        run_toml(tomlfile,opt_set)