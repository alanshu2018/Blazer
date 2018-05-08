#coding: utf-8

import numpy as np
from hyperopt import fmin, hp, tpe
import hyperopt
from time import clock

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import datasets as ds
from sklearn.model_selection import KFold
import sys


class Tunner(object):
    def __init__(self):
        self.best_step = 0 #best iterations

    def refine_args(self, args):
        return args

    def train_and_evaluate(self, model,config):
        raise NotImplementedError

    def build(self):
        raise NotImplementedError

    def get_space(self):
        raise NotImplementedError

    def tune(self):
        def objective(args):
            model,config = self.build(args)
            metric,step = self.train_and_evaluate(model,config)
            self.best_step = step
            return metric

        space = self.get_space()
        best_sln = fmin(objective, space, algo=tpe.suggest, max_evals=20)
        best = hyperopt.space_eval(space,best_sln)
        best = self.refine_args(best)
        #best_sln = fmin(objective, space, algo=hyperopt.anneal.suggest, max_evals=300)
        return best

