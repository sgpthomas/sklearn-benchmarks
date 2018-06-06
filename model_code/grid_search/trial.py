from pmlb import classification_dataset_names
import os
import sys

import AdaBoostClassifier
import BernoulliNB
import DecisionTreeClassifier
import ExtraTreesClassifier
import GaussianNB
import GradientBoostingClassifier
import KNeighborsClassifier
import LinearSVC
import LogisticRegression
import MultinomialNB
import PassiveAggressiveClassifier
import RandomForestClassifier
import SGDClassifier
import SVC
import XGBClassifier

datasets = classification_dataset_names[:1]

resultdir = sys.argv[1]

try:
    os.mkdir(resultdir)
except OSError:
    print("Results directory already exists.")
    exit(-1)

for name in datasets:
    AdaBoostClassifier.run(name, resultdir=resultdir)
    BernoulliNB.run(name, resultdir=resultdir)
    DecisionTreeClassifier.run(name, resultdir=resultdir)
    ExtraTreesClassifier.run(name, resultdir=resultdir)
    GaussianNB.run(name, resultdir=resultdir)
    GradientBoostingClassifier.run(name, resultdir=resultdir)
    KNeighborsClassifier.run(name, resultdir=resultdir)
    LinearSVC.run(name, resultdir=resultdir)
    LogisticRegression.run(name, resultdir=resultdir)
    MultinomialNB.run(name, resultdir=resultdir)
    PassiveAggressiveClassifier.run(name, resultdir=resultdir)
    RandomForestClassifier.run(name, resultdir=resultdir)
    SGDClassifier.run(name, resultdir=resultdir)
    SVC.run(name, resultdir=resultdir)
    XGBClassifier.run(name, resultdir=resultdir)
