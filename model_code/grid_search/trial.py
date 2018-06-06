import AdaBoostClassifier
from pmlb import classification_dataset_names
import os
import sys

datasets = classification_dataset_names[:1]

resultdir = sys.argv[1]

try:
    os.mkdir(resultdir)
except OSError:
    print("Results directory already exists.")
    exit(-1)

for name in datasets:
    AdaBoostClassifier.run(name, resultdir=resultdir)
