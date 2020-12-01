#!/usr/bin/env python
import torch
import pandas as pd

#df = pd.read_csv("/Users/shalin/Desktop/ASSIStments2005.tsv", sep='\t')
#print(df)

device = torch.device("cuda:0")

print("Loading CSV File...")
test = pd.read_csv("/Users/shalin/Desktop/2005_2006.tsv", sep='\t')
print(test)
test_tensor = torch.tensor(test.values)
print(test_tensor)


