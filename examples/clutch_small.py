from model import AutomatedInnovization

import pandas as pd

dataset = pd.read_csv("datasets/clutch.csv")

rules = [["ro", "ri"], ["T", "S"]]

for rule in rules:
    AutomatedInnovization(dataset=dataset, basis_functions=rule, drop_duplicates=True).findRules()
