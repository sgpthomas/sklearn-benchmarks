import sys
import itertools
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from tpot_metrics import balanced_accuracy_score
import warnings
from pmlb import fetch_data

def map_dict(f, d):
    return { k: f(v) for k, v in d.items() }

def evaluate_model(dataset, pipeline_components, pipeline_parameters, resultdir="."):
    # input_data = pd.read_csv(dataset, compression='gzip', sep='\t')
    input_data = fetch_data(dataset, local_cache_dir="../../pmlb-cache")
    features = input_data.drop('target', axis=1).values.astype(float)
    labels = input_data['target'].values

    pipelines = [dict(zip(pipeline_parameters.keys(), list(parameter_combination)))
                 for parameter_combination in itertools.product(*pipeline_parameters.values())]

    header_text = '\t'.join(["dataset", "classifier",
                                 "parameters", "test_accuracy",
                                 "train_accuracy", "test_f1_macro", "train_f1_macro"])
    results_dict = { "dataset": [],
                     "classifier": [],
                     "parameters": [],
                     "test_accuracy": [],
                     "train_accuracy": [],
                     "test_f1_macro": [],
                     "train_f1_macro": [] }
    with warnings.catch_warnings():
        # Squash warning messages. Turn this off when debugging!
        warnings.simplefilter('ignore')

        for pipe_parameters in pipelines:
            pipeline = []
            for component in pipeline_components:
                if component in pipe_parameters:
                    args = pipe_parameters[component]
                    pipeline.append(component(**args))
                else:
                    pipeline.append(component())

            try:
                clf = make_pipeline(*pipeline)
                cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=90483257)
                scoring = ['accuracy', 'f1_macro']
                validation = cross_validate(clf, features, labels, cv=cv, scoring=scoring)
                avg = map_dict(lambda v: np.mean(v), validation)
                # balanced_accuracy = balanced_accuracy_score(labels, cv_predictions)
            except KeyboardInterrupt:
                sys.exit(1)
            # This is a catch-all to make sure that the evaluation won't crash due to a bad parameter
            # combination or bad data. Turn this off when debugging!
            # except Exception as e:
            #     continue

            classifier_class = pipeline_components[-1]
            param_string = ','.join(['{}={}'.format(parameter, value)
                                    for parameter, value in pipe_parameters[classifier_class].items()])

            out_text = '\t'.join([dataset,
                                  classifier_class.__name__,
                                  param_string,
                                  str(avg['test_accuracy']),
                                  str(avg['train_accuracy']),
                                  str(avg['test_f1_macro']),
                                  str(avg['train_f1_macro'])])
            print(out_text)
            sys.stdout.flush()

            results_dict["dataset"].append(dataset)
            results_dict["classifier"].append(classifier_class.__name__)
            results_dict["parameters"].append(param_string)
            results_dict["test_accuracy"].append(avg["test_accuracy"])
            results_dict["train_accuracy"].append(avg["test_accuracy"])
            results_dict["test_f1_macro"].append(avg["test_f1_macro"])
            results_dict["train_f1_macro"].append(avg["train_f1_macro"])

            fn = '{}/tmp-{}-{}.pkl'.format(resultdir, dataset, classifier_class.__name__)
            pd.DataFrame(results_dict).to_pickle(fn)

        fn = '{}/final-{}-{}.pkl'.format(resultdir, dataset, classifier_class.__name__)
        pd.DataFrame(results_dict).to_pickle(fn)
