import importlib

class TestcaseLoad:
    @staticmethod
    def load_testcase(testcase_key,testcase_dict):
        testcase_module = importlib.import_module("testcases."+testcase_key)
        return getattr(testcase_module, testcase_dict["testfunc"])