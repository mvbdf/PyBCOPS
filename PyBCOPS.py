import numpy as np
import pandas as pd


def conformalscore(ste, s):
    """ Adaptado de conformalscore.cpp """
    return [np.sum(ste[i] >= s) for i in range(len(ste))]


def conformal_scores(ste, s, y, labels):
    K = len(labels)
    prediction_te = np.zeros((np.shape(ste)[0], K))

    for k in range(K):
        sk = s[y == labels[k], k]

        prediction_te[:,k] = conformalscore(ste[:,k], sk)
        prediction_te[:,k] = (prediction_te[:,k] + 1) / (len(sk) + 1)
    
    return prediction_te


def train(classifier, x_train, y_train, labels, x_test, *classifier_args, **classifier_kargs):
    K = len(labels)
    models = [None]*K

    for k in range(K):
        xk = np.concatenate((x_test, x_train[y_train == labels[k],:]))
        yk = np.concatenate((np.repeat(0, np.shape(x_test)[0]),
                             np.repeat(1, np.sum(y_train == labels[k]))))
        
        models[k] = classifier(*classifier_args, **classifier_kargs).fit(xk, yk)

    return models


def prediction(models, x_train, y_train, labels, x_test):
    K = len(labels)
    s = np.zeros((len(y_train), K))
    ste = np.zeros((np.shape(x_test)[0], K))

    for k in range(K):    
        if np.sum(y_train == labels[k]) > 0:
            temp1 = models[k].predict_proba(x_train)
            temp2 = models[k].predict_proba(x_test)

            try:
                temp3 = np.shape(temp1)[1]
            except IndexError:
                s[:,k] = temp1
                ste[:,k] = temp2
            else:
                s[:,k] = temp1[:,temp3-1]
                ste[:,k] = temp2[:,temp3-1]

    prediction_conformal = conformal_scores(ste, s, y_train, labels)
    return {'prediction_conformal':prediction_conformal, 'score_test':ste, 'score_train':s}


def BCOPS(classifier, x_train1, y_train1, x_train2, y_train, x_test1, x_test2,
          labels, *classifier_args, **classifier_kargs):
    # Training
    models1 = train(classifier, x_train2, y_train, labels, x_test2, *classifier_args, **classifier_kargs)
    models2 = train(classifier, x_train1, y_train1, labels, x_test1, *classifier_args, **classifier_kargs)
    
    # Prediction
    prediction1 = prediction(models1, x_train1, y_train1, labels, x_test1)['prediction_conformal']
    prediction2 = prediction(models2, x_train2, y_train, labels, x_test2)['prediction_conformal']
    return {'conformal_scores1':prediction1, 'conformal_scores2':prediction2}


def evaluate_conformal(prediction, y_test, labels, alpha=0.05):
    labels_te = np.unique(y_test)
    res = np.zeros((len(labels_te), len(labels)))

    for i in range(len(labels_te)):
        ii = np.where(y_test == labels_te[i])
        res[i,:] = np.apply_along_axis(np.mean, 1, prediction[ii,:] >= alpha)
    res = pd.DataFrame(res)

    res.columns = labels
    res.index = labels_te
    return res
