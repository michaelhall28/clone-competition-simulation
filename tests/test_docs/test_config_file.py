
from clone_competition_simulation import Parameters, Algorithm
import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
LOCAL_CONFIG_FILE = os.path.join(DIR_PATH, "..", "..", "example_run_config.yml")


def test_config():
    
    p = Parameters(run_config_file=LOCAL_CONFIG_FILE)


def test_config1():
    os.environ["CCS_RUN_CONFIG"] = LOCAL_CONFIG_FILE
    p =  Parameters()
    del os.environ["CCS_RUN_CONFIG"]


def test_config2():

    p = Parameters(run_config_file=LOCAL_CONFIG_FILE)
    assert p.algorithm == Algorithm.WF


def test_config3():

    p = Parameters(run_config_file=LOCAL_CONFIG_FILE, algorithm="Moran")

    assert p.algorithm == Algorithm.MORAN