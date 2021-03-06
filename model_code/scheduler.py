#!/usr/bin/env python3

import socket
import argparse
import trial_msg
import itertools
from pathlib import Path
import traceback
import os
import pandas as pd

from pmlb import classification_dataset_names
from todo import TodoList

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

# todos = {}
todos = TodoList()
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

def commit_result(res, ident):
    pd.DataFrame(res).to_pickle("{}/tmp-{}.pkl".format(resultdir, ident))

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

def handle_client(clients, key):
    try:
        c = clients[key]
        data = trial_msg.deserialize(c['client'].recv(trial_msg.SIZE))

        # check if client has died
        if data == None:
            print("Received 'None' from {}. Marked for removal!".format(c['client'].getsockname()))
            return 1

        # switch based on msg_type
        msg_type = data['msg_type']
        if msg_type == trial_msg.VERIFY:
            send_msg(c['client'], {'msg_type': trial_msg.SUCCESS})

        elif msg_type == trial_msg.TRIAL_REQUEST:
            ident, task = None, None
            item = todos.next()

            if item == None:
                print("No more trials!")
                send_msg(c['client'], {'msg_type': trial_msg.TERMINATE})
                return

            ident, task = item
            dataset, method, params = task
            c['task'] = ident
            send_msg(c['client'], {'msg_type': trial_msg.TRIAL_DETAILS,
                                   'dataset': dataset,
                                   'method': method,
                                   'params': params})
            print("Handed out #{}: {} {} {}".format(ident, dataset, method, params))

        elif msg_type == trial_msg.TRIAL_DONE:
            print("{} finished!".format(c['client'].getpeername()))
            size = data['size']
            send_msg(c['client'], {'msg_type': trial_msg.TRIAL_SEND})
            trial_result = trial_msg.deserialize(c['client'].recv(size))
            commit_result(trial_result, c['task'])
            todos.complete(c['task'])
            print("Commited #{} (remaining: {})".format(c['task'], len(todos.remaining())))
            c['task'] = None

        elif msg_type == trial_msg.TRIAL_CANCEL:
            print("{} aborted!".format(c['client'].getpeername()))
            todos.cancel(c['task'])
            send_msg(c['client'], {'msg_type': trial_msg.SUCCESS})
            print("Aborting #{} (remaining: {})".format(c['task'], len(todos.remaining())))
            c['task'] = None

        elif msg_type == trial_msg.TERMINATE:
            return 1

        else:
            print("Unknown msg type: {}!".format(msg_type))
            send_msg(c['client'], {'msg_type': trial_msg.INVALID})
    except socket.timeout:
        pass
    except Exception as e:
        print("There was an exception while handling {}".format(c['client'].getsockname()))
        traceback.print_exc()

if __name__ == "__main__":
    options = make_parser().parse_args()

    try:
        os.mkdir(options.resultdir)
        # print(options.resultdir)
    except OSError:
        pass

    resultdir = options.resultdir

    # gather todos
    print("Gathering todos...", end='', flush=True)
    if options.default:
        params = {key: [{}] for key in classifier_functions}
    else:
        params = {key: item.get_pipeline_parameters()
                  for key, item in classifier_functions.items()}
    for name in classification_dataset_names:
        for key in params:
            for x in params[key]:
                todos.add((name, key, x))

    if options.resume:
        p = Path(resultdir)
        ids = list(map(lambda s: int(s.stem.split('-')[1]), p.glob("tmp-*.pkl")))
        for i in ids:
            todos.complete(i)

    total = todos.size()
    print("found {} incomplete trials!".format(len(todos.remaining())))

    clients = {}
    s = start_server(options.port)
    s.settimeout(1)
    print("Starting to listen...")
    s.listen(options.max_connections)
    while len(todos.completed()) != todos.size():
        try:
            to_remove = []
            for k in clients:
                if handle_client(clients, k) != None:
                    to_remove.append(k)
            for item in to_remove:
                print("Removing {}".format(item))
                if clients[item]['task'] != None:
                    todos.abort(clients[item]['task'])
                del clients[item]

            c, addr = s.accept()
            c.settimeout(1)
            clients[addr] = {'client': c, 'task': None}
            print("Connected to {}:{}!".format(addr[0], addr[1]))
        except socket.timeout:
            pass
        except KeyboardInterrupt:
            print("Shutting down server!")
            print("Aborting progress on:")
            for item in todos.in_progress():
                print(" [-] {}".format(item))
            s.close()
            exit(0)
    print("No more todos!")
    for item in todos.in_progress():
        print(" [-] {}".format(item))
