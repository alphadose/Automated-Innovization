from model import AutomatedInnovization

import pandas as pd

dataset = pd.read_csv("../datasets/spring.csv")

rules = [["D", "N"], ["d", "D"], ["d", "D", "N"]]

for rule in rules:
    AutomatedInnovization(dataset=dataset, basis_functions=rule).findRules()
