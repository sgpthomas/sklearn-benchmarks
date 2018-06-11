from pmlb import classification_dataset_names
import os
import sys
import glob
import itertools
import argparse
import multiprocessing
import traceback

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

classifier_functions = {
    "AdaBoostClassifier": AdaBoostClassifier.run,
    "BernoulliNB": BernoulliNB.run,
    "DecisionTreeClassifier": DecisionTreeClassifier.run,
    "ExtraTreesClassifier": ExtraTreesClassifier.run,
    "GaussianNB": GaussianNB.run,
    "GradientBoostingClassifier": GradientBoostingClassifier.run,
    "KNeighborsClassifier": KNeighborsClassifier.run,
    "LinearSVC": LinearSVC.run,
    "LogisticRegression": LogisticRegression.run,
    "MultinomialNB": MultinomialNB.run,
    "PassiveAggressiveClassifier": PassiveAggressiveClassifier.run,
    "RandomForestClassifier": RandomForestClassifier.run,
    "SGDClassifier": SGDClassifier.run,
    "SVC": SVC.run,
    "XGBClassifier": XGBClassifier.run
}

def run(resultdir, dataset, method):
    if not isinstance(dataset, str):
        raise TypeError("Input must be a string!")

    classifier_functions[method](dataset, resultdir=resultdir, use_params=False)

def completed(resultdir, include_in_progress=True):
    files = []
    if include_in_progress:
        files = glob.glob("{}/*.pkl".format(resultdir))
    else:
        files = glob.glob("{}/final-*.pkl".format(resultdir))
    rmext = lambda s: s[:-4]
    f = lambda s: tuple((s.split('/')[1]).split('--')[1:3])
    return list(map(f, map(rmext, files)))

def make_parser():
    dscr = "Script to run sklearn benchmarks on the PMLB datasets."
    parser = argparse.ArgumentParser(description=dscr)

    parser.add_argument('resultdir', action="store")
    parser.add_argument('-t', '--temp', action="store_false")
    parser.add_argument('-d', '--dataset', action="store_true")
    parser.add_argument('-c', '--count', action="store", default=1, type=int)
    parser.add_argument('-p', '--processes', action="store", default=1, type=int)

    return parser


if __name__ == "__main__":
    options = make_parser().parse_args()

    try:
        os.mkdir(options.resultdir)
    except OSError:
        pass

    def iteration(todo, attempts=5):
        try:
            dataset, method = todo
            print(todo)
            run(options.resultdir, dataset, method)
        except Exception as err:
            if attempts < 0:
                traceback.print_exception(err)
                return None
            else:
                print("Tring {} again!".format(todo))
                iteration(todo, attempts=(attempts-1))

    f = lambda x: x not in completed(options.resultdir, include_in_progress=options.temp)
    zipped = list(itertools.product(classification_dataset_names,
                                    list(classifier_functions.keys())))
    todos = list(filter(f, zipped))
    if todos == []:
        print("No more datasets or methods!")
        exit(-1)

    if options.dataset:
        options.count *= len(classifier_functions.keys())

    if options.count > len(todos):
        options.count = len(todos)

    with multiprocessing.Pool(processes=options.processes) as pool:
        list(pool.map(iteration, todos[:options.count]))

    print("Done!")
