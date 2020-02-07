from model import AutomatedInnovization

import pandas as pd

dataset = pd.read_csv("../datasets/truss.csv")

rules = [["S", "V"], ["x1", "V"], ["x2", "V"], ["x2", "x1"]]

for rule in rules:
    AutomatedInnovization(dataset=dataset, basis_functions=rule).findRules()
