#coding: utf-8

from Learner import *
from DataReader import *
from utils import dist_utils
import config

from LearnerManager import *

class Task(object):
    learner_manager = LearnerManager()

    def __init__(self,name, data_conf, learner_name, logger, verbose=True, plot_importance=False, refit_once=False):
        self.name = name
        self.logger = logger
        self.data_config = data_conf
        self.learner_name = learner_name
        self.n_fold =5
        self.plot_importance = plot_importance
        self.refit_once = refit_once

        self.data_loader = self._get_data_loader(data_conf)
        self.learner = self._get_learner(learner_name)
        if self.learner is None:
            raise Exception("Unknown leaner:{}".format(learner_name))
        self.verbose = verbose
        self.rmse_cv_mean, self.rmse_cv_std = 0.0, 0.0

    def _get_data_loader(self,data_conf):
        loader = DataReader(data_conf, self.logger, self.n_fold)
        return loader

    def _get_learner(self, name):
        learner = self.learner_manager.get_learner(name)
        return learner

    def __str__(self):
        feature_name = self.data_config.name
        return "Feat@%s_Learner@%s"%(feature_name, self.learner.name)

    def _print_param_dict(self, d, prefix="      ", incr_prefix="      "):
        for k,v in sorted(d.items()):
            if isinstance(v, dict):
                self.logger.info("%s%s:" % (prefix,k))
                self._print_param_dict(v, prefix+incr_prefix, incr_prefix)
            else:
                self.logger.info("%s%s: %s" % (prefix,k,v))

    def simple_validate(self, params_dict, X, y):
        self.logger.info("Simple CV with parms:{}".format(params_dict))
        start = time.time()
        if self.verbose:
            self.logger.info("="*50)
            self.logger.info("Task")
            self.logger.info("      %s" % str(self.__str__()))
            self.logger.info("Param")
            self._print_param_dict(params_dict)
            self.logger.info("Result")
            self.logger.info("      Run      RMSE        Shape")

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.10, random_state=1024)

        # data
        #X_train, y_train, X_valid, y_valid = self.feature._get_train_valid_data(i)
        model = self.learner.create_model(params_dict)
        # fit
        self.logger.info("Fit for train data")
        model = self.learner.fit(model,X_train, y_train)
        self.logger.info("Predict for validation data")
        y_pred = np.reshape(self.learner.predict(model,X_test),(len(X_test),))
        print(y_test.shape)
        print(y_pred.shape)
        auc_cv = dist_utils._auc(y_test, y_pred)
        # log
        self.logger.info("      {:>3}    {:>8}    {} x {}".format(
            1, np.round(auc_cv,6), X_train.shape[0], X_train.shape[1]))

        del(model)
        del(y_pred)
        gc.collect()

        self.rmse_cv_mean_simple = 1- auc_cv
        self.rmse_cv_std_simple = 0
        end = time.time()
        _sec = end - start
        _min = int(_sec/60.)
        if self.verbose:
            self.logger.info("AUC")
            self.logger.info("      Mean: %.6f"%self.rmse_cv_mean)
            self.logger.info("      Std: %.6f"%self.rmse_cv_std)
            self.logger.info("Time")
            if _min > 0:
                self.logger.info("      %d mins"%_min)
            else:
                self.logger.info("      %d secs"%_sec)
            self.logger.info("-"*50)
        return self

    def cv(self, params_dict, sample_len=0, do_save = False):
        self.logger.info("CV with parms:{}".format(params_dict))
        start = time.time()
        if self.verbose:
            self.logger.info("="*50)
            self.logger.info("Task")
            self.logger.info("      %s" % str(self.__str__()))
            self.logger.info("Param")
            self._print_param_dict(params_dict)
            self.logger.info("Result")
            self.logger.info("      Run      RMSE        Shape")

        auc_cv = np.zeros(self.n_fold)
        total_train = self.data_loader.len_train
        train_pred = np.zeros(total_train)
        i = -1
        cnt = 0
        y_pred_test = None
        X_test, y_test,W_test = self.data_loader._get_test_data()
        for (X_train, y_train, X_valid, y_valid,train_ind,valid_ind,W_train,W_valid) in self.data_loader._get_train_valid_data():
            i += 1
            cnt += 1
            print(X_train[:10])
            print(y_train[:10])
            cur_train_len = len(X_train)
            if sample_len >0 and cur_train_len > sample_len:
                X_train = X_train[-sample_len:]
                y_train = y_train[-sample_len:]

            # data
            #X_train, y_train, X_valid, y_valid = self.feature._get_train_valid_data(i)
            model = self.learner.create_model(params_dict)
            # fit
            model = self.learner.fit(model,X_train, y_train)
            self.logger.info("Predict for validation data")
            y_pred = np.reshape(self.learner.predict(model,X_valid),(len(X_valid),))
            train_pred[valid_ind] = y_pred
            if do_save and X_test is not None:
                self.logger.info("Predict for test data")
                if y_pred_test is None:
                    y_pred_test = np.reshape(self.learner.predict(model,X_test),(len(X_test),))
                else:
                    y_pred_test += np.reshape(self.learner.predict(model,X_test),(len(X_test),))

            print(y_valid.shape)
            print(y_pred.shape)
            auc_cv[i] = dist_utils._auc(y_valid, y_pred)
            # log
            self.logger.info("      {:>3}    {:>8}    {} x {}".format(
                i+1, np.round(auc_cv[i],6), X_train.shape[0], X_train.shape[1]))

            del(model)
            del(y_pred)
            gc.collect()

            if sample_len > 0 and do_save == False: # with sample we only do some evaluation
                break

        # save
        if do_save:
            fname = "%s/cv_pred.%s.csv"%(config.OUTPUT_DIR, self.__str__())
            self.logger.info("Save cv predictions:{}".format(fname))
            df = pd.DataFrame({"predicted": train_pred})
            df.to_csv(fname, index=False, columns=["predicted"])

            if y_pred_test is not None:
                fname = "%s/cv_test_avg_pred.%s.csv"%(config.OUTPUT_DIR, self.__str__())
                self.logger.info("Save cv test predictions:{}".format(fname))
                df = pd.DataFrame({"click_id":y_test,"predicted": y_pred_test})
                df.to_csv(fname, index=False, columns=["predicted"])

        #self.rmse_cv_mean = 1- np.mean(auc_cv)
        # auc_error = 1 - auc
        self.rmse_cv_mean = 1- np.mean(auc_cv[:cnt])
        self.rmse_cv_std = np.std(auc_cv[:cnt])
        end = time.time()
        _sec = end - start
        _min = int(_sec/60.)
        if self.verbose:
            self.logger.info("AUC")
            self.logger.info("      Mean: %.6f"%self.rmse_cv_mean)
            self.logger.info("      Std: %.6f"%self.rmse_cv_std)
            self.logger.info("Time")
            if _min > 0:
                self.logger.info("      %d mins"%_min)
            else:
                self.logger.info("      %d secs"%_sec)
            self.logger.info("-"*50)
        return self

    def refit(self, params_dict, do_save=True):
        self.logger.info("Refit with parms:{}".format(params_dict))
        model = self.learner.create_model(params_dict)
        X_train, y_train, X_test, y_test, W_train, W_test= self.data_loader._get_train_test_data()
        len_test = len(X_test)
        print("len_test={}".format(len(X_test)))
        print("len_train={}".format(len(X_train)))
        """
        for i in range(self.n_fold):
            if self.refit_once and i >= 1:
                break
            self.learner.fit(model, X_train, y_train)
            if i == 0:
                y_pred = np.reshape(self.learner.predict(model,X_test),(len_test,))
            else:
                y_pred += np.reshape(self.learner.predict(model,X_test),(len_test,))
        if not self.refit_once:
            y_pred /= float(self.n_fold)

        """
        if self.plot_importance:
            feature_names = self.data_loader._get_feature_names()
            model = self.learner.fit(model,X_train, y_train, feature_names)
            y_pred = self.learner.predict(model,X_test, feature_names)
        else:
            model = self.learner.fit(model,X_train, y_train)
            y_pred = self.learner.predict(model, X_test)

        y_pred = np.reshape(y_pred,(len_test,))
        # save
        # submission
        if do_save:
            fname = "%s/sub_pred.%s.[Mean%.6f]_[Std%.6f].csv"%(
					config.SUBM_DIR, self.__str__(), self.rmse_cv_mean, self.rmse_cv_std)
            print(y_test.shape)
            print(y_pred.shape)
            print("Writing to file:{}".format(fname))
            pd.DataFrame({"click_id": y_test, "is_attributed": y_pred}).to_csv(fname, index=False)

        # plot importance
        if self.plot_importance:
            ax = self.learner.plot_importance()
            ax.figure.savefig("%s/%s.pdf"%(config.FIG_DIR, self.__str__()))
        return self

    def go(self,params_dict, sample_len=0, do_save_cv=False,do_save_refit=True):
        self.cv(params_dict,sample_len, do_save_cv)
        self.refit(params_dict,do_save_refit)
        return self


class TaskOptimizer(object):
    def __init__(self, name, learner_name, data_config, logger,
                 max_evals=100, verbose=True, refit_once=False, plot_importance=False):
        self.learner_name = learner_name
        self.data_config = data_config
        self.logger = logger
        self.max_evals = max_evals
        self.verbose = verbose
        self.refit_once = refit_once
        self.plot_importance = plot_importance
        self.trial_counter = 0

        self.task = Task(name, data_config,learner_name,logger,verbose,plot_importance,refit_once)

    def _obj(self, param_dict):
        self.trial_counter += 1

        param_dict = self.task.learner._convert_int_param(param_dict)
        #self.task.go(param_dict,sample_len= 100000,do_save_cv=False,do_save_refit=False)
        sample_len = 10000000
        self.logger.info("==>Tunne with CV")
        #self.task.cv(param_dict,sample_len= sample_len,do_save=False)
        predictors = self.task.data_loader._get_feature_names()
        X_train, y_train, W_train = self.task.data_loader.get_sub_data(predictors,sample_ratio = 0.1)
        self.task.simple_validate(param_dict,X_train,y_train)

        ret = {
            "loss": self.task.rmse_cv_mean,
            "attachments": {
                "std": self.task.rmse_cv_std,
            },
            "status": STATUS_OK,
        }
        return ret

    def run(self):
        self.logger.info("First tune the model to Find the Best solution")
        start = time.time()
        trials = Trials()
        params_space = self.task.learner.param_space
        best = fmin(self._obj, params_space, tpe.suggest, self.max_evals, trials)
        best_params = space_eval(params_space, best)
        best_params = self.task.learner._convert_int_param(best_params)
        trial_rmses = np.asarray(trials.losses(), dtype=float)
        best_ind = np.argmin(trial_rmses)
        best_rmse_mean = trial_rmses[best_ind]
        best_rmse_std = trials.trial_attachments(trials.trials[best_ind])["std"]
        self.logger.info("-"*50)
        self.logger.info("Best AUC Error")
        self.logger.info("      Mean: %.6f"%best_rmse_mean)
        self.logger.info("      std: %.6f"%best_rmse_std)
        self.logger.info("Best param")
        self.task._print_param_dict(best_params)
        end = time.time()
        _sec = end - start
        _min = int(_sec/60.)
        self.logger.info("Time")
        if _min > 0:
            self.logger.info("      %d mins"%_min)
        else:
            self.logger.info("      %d secs"%_sec)
        self.logger.info("-"*50)

        self.logger.info("Rerun the alg to get the CV and Final submission")
        self.rerun(best_params)


    def rerun(self,params_dict):
        #self.task.go(params_dict,do_save_cv=True, do_save_refit=True)
        self.task.refit(params_dict,do_save=True)

    def select_features(self, given_predictors):
        """
        Select subset data features
        :param given_predictors:
        :return:
        """
        predictors = self.task.data_loader._get_feature_names()
        try_predictors = [ feature for feature in predictors if feature not in given_predictors]
        good_predictors = [ p for p in given_predictors]
        self.logger.info("===========>Try feature good_predictors={}".format(good_predictors))
        self.logger.info("===========>Try feature try_predictors={}".format(try_predictors))
        last_rmse_mean, last_rmse_std = self._try_features(good_predictors)
        max_tries = int(len(try_predictors) * 1.25)
        tries = 0
        while len(try_predictors) > 0 and tries < max_tries:
            tries += 1
            to_add = np.random.choice(try_predictors)
            cur_predictors = good_predictors + [to_add]
            self.logger.info("===========>Try feature :tries={}, {}, cur={}".format(tries,to_add, cur_predictors))
            rmse_mean, rmse_std = self._try_features(cur_predictors)
            self.logger.info("===========>Last rmse_mean={}, rmse_mean={}".format(last_rmse_mean,rmse_mean))
            if rmse_mean < last_rmse_mean:
                self.logger.info("===========>Add feature {}, cur={}".format(to_add, cur_predictors))
                good_predictors.append(to_add)
                try_predictors.remove(to_add)
                last_rmse_mean = rmse_mean
            else:
                self.logger.info("===========>Reject feature {}, cur={}".format(to_add, cur_predictors))

        self.logger.info("Final good predictors:{}".format(good_predictors))
        print("Final good predictors:{}".format(good_predictors))
        return good_predictors

    def _obj_with_data(self, param_dict,X_train,y_train):
        param_dict = self.task.learner._convert_int_param(param_dict)
        #self.task.go(param_dict,sample_len= 100000,do_save_cv=False,do_save_refit=False)
        self.logger.info("==>Tunne with CV")
        self.task.simple_validate(param_dict,X_train,y_train)
        ret = {
            "loss": self.task.rmse_cv_mean_simple,
            "attachments": {
                "std": self.task.rmse_cv_std_simple,
            },
            "status": STATUS_OK,
        }
        return ret

    def _try_features(self, cur_predictors):
        # Get a subset of data
        X_train, y_train, W_train = self.task.data_loader.get_sub_data(cur_predictors,sample_ratio = 0.05)
        self.logger.info("Try features : data_len={}".format(len(X_train)))

        start = time.time()
        trials = Trials()
        params_space = self.task.learner.param_space
        best = fmin(lambda params_dict: self._obj_with_data(params_dict,X_train,y_train),
                    params_space, tpe.suggest, 10, trials)
        best_params = space_eval(params_space, best)
        best_params = self.task.learner._convert_int_param(best_params)
        trial_rmses = np.asarray(trials.losses(), dtype=float)
        best_ind = np.argmin(trial_rmses)
        best_rmse_mean = trial_rmses[best_ind]
        best_rmse_std = trials.trial_attachments(trials.trials[best_ind])["std"]
        self.logger.info("-"*50)
        self.logger.info("Best AUC Error")
        self.logger.info("      Mean: %.6f"%best_rmse_mean)
        self.logger.info("      std: %.6f"%best_rmse_std)
        self.logger.info("Best param")
        self.task._print_param_dict(best_params)
        end = time.time()
        _sec = end - start
        _min = int(_sec/60.)
        self.logger.info("Time")
        if _min > 0:
            self.logger.info("      %d mins"%_min)
        else:
            self.logger.info("      %d secs"%_sec)
        self.logger.info("-"*50)
        return best_rmse_mean,best_rmse_std
