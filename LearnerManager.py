from Learner import *

from MyKerasLearner import *
from LGBMLearner import *
#from FMFtrlLearner import *
#from DeepFMRankNetLearner1 import *
#from DeepFMNetLearner import *
#from DeepLRLearner import *
class LearnerManager(object):
    learners = [
        LGBMClassifierLearner(),
        ####### Regressors ##############
        LassoRegressorLearner(),
        RidgeRegressorLearner(),
        RandomRidgeRegressorLearner(),
        BayesianRidgeRegressorLearner(),
        LinearSVRRegressorLearner(),
        SVRRegressorLearner(),
        ExtraTreeRegressorLearner(),
        RGFRegressorLearner(),
        KerasDNNRegressorLearner(),
        AdaBoostRegressorLearner(),
        GBMRegressorLearner(),
        RandomForestRegressorLearner(),
        MyKerasDNNRegressorLearner(),
        LightGBMLearner(),
        #DeepFMRankLearner(),
        #DeepFMLearner(),
        #DeepLRLearner(),
        #FMFtrlLearner(),
    ]
    def __init__(self):
        for i in range(len(self.learners)):
            self.learners[i].name = self.learners[i].name.strip()
            print("Learner:{}".format(self.learners[i].name))

    def get_learner(self,name):
        name = name.strip()
        for learner in self.learners:
            if learner.name == name:
                return learner

        return None

