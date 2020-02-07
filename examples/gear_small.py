from model import AutomatedInnovization

import pandas as pd

dataset = pd.read_csv("../datasets/gear.csv")

rules = [["Td", "Tb"], ["Ta", "Tf"], ["Tf/Tb", "Ta/Td"]]

for rule in rules:
    AutomatedInnovization(dataset=dataset, basis_functions=rule).findRules()
