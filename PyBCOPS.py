import numpy as np
import pandas as pd

# from ctypes import CDLL, POINTER, c_int, c_float


# libscore = CDLL("./libscore.so")
# libscore.conformalscore.argtypes = [
#     POINTER(c_float),
#     POINTER(c_float),
#     POINTER(c_int),
#     c_int,
#     c_int
# ]

# def conformalscore(s_test, s_train):
#     size_ste = len(s_test)
#     size_s = len(s_train)
#     ste = (c_float * size_ste)(*s_test)
#     s = (c_float * size_s)(*s_train)
#     count = (c_int * size_ste)()
#     libscore.conformalscore(ste, s, count, size_ste, size_s)
#     return np.array(count)


def conformalscore(s_test, s_train):
    """ Adapted from conformalscore.c but pretty much as fast as the C version """
    return [np.sum(i >= s_train) for i in s_test]


def conformal_scores(s_test, s_train, y_train, labels):
    """ Conformal prediction for any given scores """
    prediction_test = np.zeros((np.shape(s_test)[0], np.shape(labels)[0]))

    for k, label in enumerate(labels):
        sk = s_train[y_train == label, k]

        prediction_test[:,k] = conformalscore(s_test[:,k], sk)
        prediction_test[:,k] = (prediction_test[:,k] + 1) / (len(sk) + 1)
    
    return prediction_test


def train(classifier, x_train, y_train, labels, x_test, *classifier_args, **classifier_kargs):
    """ 
    Returns a list of trained models that are used to predict the
    BCOPs score for each new observation
    """
    models = [None]*len(labels)

    for k, label in enumerate(labels):
        xk = np.concatenate((x_test, x_train[y_train == label,:]))
        yk = np.concatenate((np.repeat(0, np.shape(x_test)[0]),
                             np.repeat(1, np.sum(y_train == label))))
        
        models[k] = classifier(*classifier_args, **classifier_kargs).fit(xk, yk)

    return models


def prediction(models, x_train, y_train, labels, x_test):
    """
    Returns 
    -------
    prediction_conformal : a m by K matrix for m test samples, it is the
    conformal constructed p-value for a test sample not from each of the
    K classes. If we want to control the type I error at alpha, then, we
    assign all class labels whose conformal p-value is no smaller than alpha
    to the test samples.

    scores_test : a m by K matrix for m test samples and K classes, each entry is
    the value evaluated at a test sample using score function for a training class.

    scores_train : a n by K matrix for n training samples, each entry is the
    value evaluated at a training sample using score function for a training class.
    """
    K = len(labels)
    s = np.zeros((len(y_train), K))
    ste = np.zeros((np.shape(x_test)[0], K))

    for k, model in enumerate(models):
        if np.sum(y_train == labels[k]) > 0:
            temp1 = model.predict_proba(x_train)
            temp2 = model.predict_proba(x_test)

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


def BCOPS(classifier, X_train, y_train, X_test, *classifier_args, **classifier_kargs):
    """BCOPS function that does both training and prediction. 
    
    Returns
    -------
    prediction_conformal : the conformal scores for all the test observations.
    """
    
    # Data-split
    foldid = np.random.randint(1, 3, len(y_train))
    foldid_te = np.random.randint(1, 3, np.shape(X_test)[0])

    x_train1 = X_train[foldid==1,:]
    x_train2 = X_train[foldid==2,:]
    y_train1 = y_train[foldid==1]
    y_train2 = y_train[foldid==2]
    x_test1 = X_test[foldid_te ==1,:]
    x_test2 = X_test[foldid_te==2,:]
    labels = np.unique(y_train)

    # Training
    models1 = train(classifier, x_train2, y_train2, labels, x_test2, *classifier_args, **classifier_kargs)
    models2 = train(classifier, x_train1, y_train1, labels, x_test1, *classifier_args, **classifier_kargs)
    
    # Prediction
    prediction1 = prediction(models1, x_train1, y_train1, labels, x_test1)['prediction_conformal']
    prediction2 = prediction(models2, x_train2, y_train2, labels, x_test2)['prediction_conformal']

    prediction_conformal = np.zeros((np.shape(X_test)[0], len(labels)))
    prediction_conformal[foldid_te==1,:] = prediction1
    prediction_conformal[foldid_te==2,:] = prediction2

    return prediction_conformal


def evaluate_conformal(conformal_scores, y_test, labels, alpha=0.05):
    """
    Returns
    -------
    results : a result table with the columns being the classes in the test
    samples, and the rows being the classes in the training samples. The entry
    at row j and column k represents the percent of samples in class j assigned
    label k.
    """
    labels_test = np.unique(y_test)
    results = np.zeros((len(labels_test), len(labels)))

    for i, label in enumerate(labels_test):
        ii = np.where(y_test == label)
        results[i,:] = np.apply_along_axis(np.mean, 1, conformal_scores[ii,:] >= alpha)
    results = pd.DataFrame(results)

    results.columns = labels
    results.index = labels_test
    return results


def prediction_sets(conformal_scores, labels, alpha=0.05):
    """ Returns a list with the predictions sets for the test data calculated at a given alpha """ 
    pred = conformal_scores > alpha
    y_pred = [np.ndarray.tolist(labels[i]) for i in pred]

    return y_pred


def abstention_rate(y_pred, y_test, labels):
    """ Returns the abstention rate for the outlier observations """
    abstention_counter = 0
    outlier_counter = 0

    for real_class, pred in zip(y_test, y_pred):
        if real_class not in labels:
            outlier_counter += 1
            if not pred:
                abstention_counter += 1

    return abstention_counter / outlier_counter
