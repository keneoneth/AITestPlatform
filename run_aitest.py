import toml
import argparse
import json

PATH_TO_TESTRUN = "./testrun/"
PATH_TO_TESTCASE = "./testcases/"
PATH_TO_MODEL = "./models/"
PATH_TO_RESULT = "./results/"
PATH_TO_DATASET = "./datasets/"


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

    def __init__(self,dataset_key,model_key,testcase_key,out_format):
        self.dataset_key = dataset_key
        self.model_key = model_key
        self.testcase_key = testcase_key
        self.out_format = out_format
    
    def __str__(self):
        return "dataset:{}|model:{}|testcase:{}".format(self.dataset_key,self.model_key,self.testcase_key)
    
    def out_json_name(self):
        return '%s_%s.json'%(self.title,self.testcase_key)

    def run_test(self):
        from _scripts.load_dataset import DataLoad
        data = DataLoad.load_dataset(self.dataset_key,TestRun.LOADED_DATASETS[self.dataset_key])
        from _scripts.load_model import ModelLoad
        model = ModelLoad.load_model(self.model_key,TestRun.LOADED_MODELS[self.model_key])
        from _scripts.load_testcase import TestcaseLoad
        testfunc = TestcaseLoad.load_testcase(self.testcase_key,TestRun.LOADED_TESTCASES[self.testcase_key])
        rets = testfunc(data=data,model=model,testfunc=testfunc,testconfig=TestRun.LOADED_TESTCASES[self.testcase_key])
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
        
def validate_toml(fname):
    assert fname.endswith(".toml"), "[error] fname %s doesn't end with .toml" % str(fname)

def run_toml(tomlfile):
    KEY_TITLE = "title"
    KEY_DATASETS = "datasets"
    KEY_MODELS = "models"
    KEY_TESTCASES = "testcases"
    KEY_TESTRUN = "testrun"
        
    with open(tomlfile) as f:
        loaded_toml = toml.load(f)
        assert KEY_TITLE in loaded_toml
        assert KEY_DATASETS in loaded_toml
        assert KEY_MODELS in loaded_toml
        assert KEY_TESTCASES in loaded_toml
        assert KEY_TESTRUN in loaded_toml
        print("[info] detecting {} with title {}".format(f.name,loaded_toml[KEY_TITLE]))
        TestRun.set_title(loaded_toml[KEY_TITLE])
        TestRun.load_objects(loaded_toml[KEY_DATASETS],loaded_toml[KEY_MODELS],loaded_toml[KEY_TESTCASES])
        for testrun_dict in loaded_toml[KEY_TESTRUN]:
            testrun = TestRun(testrun_dict[TestRun.KEY_DATASET],testrun_dict[TestRun.KEY_MODEL],testrun_dict[TestRun.KEY_TESTCASE],testrun_dict[TestRun.KEY_OUTFORMAT])
            print("[info] running {}".format(testrun))
            testrun.run_test()
        print("[info] testrun {} with title {} done !!!".format(f.name,loaded_toml[KEY_TITLE]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--toml",required=True,help="toml files of testcase e.g. abc.toml,def.toml,ghl.toml")
    args = parser.parse_args()
    tomls_to_run = args.toml.split(",")
    
    [validate_toml(file) for file in tomls_to_run]
    for tomlfile in tomls_to_run:
        run_toml(tomlfile)