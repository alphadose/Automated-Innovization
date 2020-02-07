from model import AutomatedInnovization

import pandas as pd

dataset = pd.read_csv("../datasets/beam.csv")

rules = [["b", "D"], ["C", "D"]]

for rule in rules:
    AutomatedInnovization(dataset=dataset, basis_functions=rule).findRules()
