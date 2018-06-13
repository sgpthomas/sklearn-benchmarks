#!/usr/bin/env python3


import socket
import argparse
import trial_msg
import itertools
import glob
import traceback
import os
import pandas as pd

from pmlb import classification_dataset_names

# s.listen(5)

# while True:
#     c, addr = s.accept()

#     print("Connected to {}".format(addr))
#     c.send(b"Hello!")
#     c.close()

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

# globals
classifier_functions = {
    "AdaBoostClassifier": AdaBoostClassifier, # 28
    "BernoulliNB": BernoulliNB, # 140
    "DecisionTreeClassifier": DecisionTreeClassifier, # 280
    "ExtraTreesClassifier": ExtraTreesClassifier, # 1120
    "GaussianNB": GaussianNB, # 1
    "GradientBoostingClassifier": GradientBoostingClassifier, # 7840
    "KNeighborsClassifier": KNeighborsClassifier, # 54
    "LinearSVC": LinearSVC, # 320
    "LogisticRegression": LogisticRegression, # 240
    "MultinomialNB": MultinomialNB, # 20
    "PassiveAggressiveClassifier": PassiveAggressiveClassifier, # 44
    "RandomForestClassifier": RandomForestClassifier, # 1120
    "SGDClassifier": SGDClassifier, # 5000
    "SVC": SVC, # 1232
    "XGBClassifier": XGBClassifier # 30492
}

todos = []
resultdir = "."
count = 0
total = 0

def flatten(l):
    return [item for sublist in l for item in sublist]

def make_parser():
    descr = "Scheduler that hands out trials and collects results."
    parser = argparse.ArgumentParser(description=descr)

    parser.add_argument('resultdir', action="store")
    parser.add_argument('-p', '--port', action="store", default=3000, type=int)
    parser.add_argument('-t', '--temp', action="store_false")
    parser.add_argument('-m', '--max-connections', action="store", default=10, type=int)
    parser.add_argument('--default', action="store_true")
    parser.add_argument('--resume', action="store_true")

    return parser

# def completed(resultdir, include_in_progress=True):
#     files = []
#     if include_in_progress:
#         files = glob.glob("{}/*.pkl".format(resultdir))
#     else:
#         files = glob.glob("{}/final-*.pkl".format(resultdir))
#     rmext = lambda s: s[:-4]
#     f = lambda s: tuple((s.split('/')[1]).split('--')[1:3])
#     return list(map(f, map(rmext, files)))

def commit_result(res):
    global count
    pd.DataFrame(res).to_pickle("{}/tmp-{}.pkl".format(resultdir, count))
    count += 1

def start_server(port):
    s = socket.socket()
    host = socket.gethostname()
    s.bind((host, port))
    print("Started server at {}:{}".format(host, port))
    return s

def stop_server(server):
    server.close()

def send_msg(client, msg):
    client.send(trial_msg.serialize(msg))

def handle_client(client, address):
    try:
        data = trial_msg.deserialize(client.recv(trial_msg.SIZE))
        if data == None:
            print("Received 'None' from {}. Marked for removal!".format(address))
            return 1
        msg_type = data['msg_type']
        if msg_type == trial_msg.VERIFY:
            send_msg(client, {'msg_type': trial_msg.SUCCESS})

        elif msg_type == trial_msg.TRIAL_REQUEST:
            dataset, method, params = todos.pop(0)
            send_msg(client, {'msg_type': trial_msg.TRIAL_DETAILS,
                              'dataset': dataset,
                              'method': method,
                              'params': params})

        elif msg_type == trial_msg.TRIAL_DONE:
            print("{} finished!".format(address))
            size = data['size']
            send_msg(client, {'msg_type': trial_msg.TRIAL_SEND})
            trial_result = trial_msg.deserialize(client.recv(size))
            commit_result(trial_result)
            print("Commited {}/{}".format(count, total))

        elif msg_type == trial_msg.TERMINATE:
            return 1

        else:
            send_msg(client, {'msg_type': trial_msg.INVALID})
    except socket.timeout:
        pass
    except Exception as e:
        print("There was an exception while handling {}:{}".format(address[0], address[1]))
        traceback.print_exc()

if __name__ == "__main__":
    options = make_parser().parse_args()

    try:
        os.mkdir(options.resultdir)
        # print(options.resultdir)
    except OSError:
        pass

    resultdir = options.resultdir

    print("Gathering todos...", end='', flush=True)
    if options.default:
        params = {key: [{}] for key in classifier_functions}
    else:
        params = {key: item.get_pipeline_parameters()
                  for key, item in classifier_functions.items()}
    for name in classification_dataset_names:
        for key in params:
            for x in params[key]:
                todos.append((name, key, x))
    if options.resume:
        n = len(glob.glob("{}/*.pkl".format(resultdir)))
        todos = todos[n:]
        count = n
    total = len(todos)
    print("found {}!".format(total))

    clients = {}
    s = start_server(options.port)
    s.settimeout(1)
    print("Starting to listen...")
    s.listen(options.max_connections)
    while True:
        to_remove = []
        for k in clients:
            # print("Polling {}".format(k))
            if handle_client(clients[k], k) != None:
                to_remove.append(k)
        for item in to_remove:
            print("Removing {}".format(item))
            del clients[item]
        try:
            c, addr = s.accept()
            c.settimeout(1)
            clients[addr] = c
            print("Connected to {}:{}!".format(addr[0], addr[1]))
        except socket.timeout:
            pass
        except KeyboardInterrupt:
            print("Shutting down server!")
            s.close()
            exit(0)
