import numpy as np
import pandas as pd


def conformalscore(s_test, s_train):
    """ Adapted from conformalscore.cpp """
    return [np.sum(s_test[i] >= s_train) for i in range(len(s_test))]


def conformal_scores(s_test, s_train, y_train, labels):
    """ Conformal prediction for any given scores """
    K = len(labels)
    prediction_test = np.zeros((np.shape(s_test)[0], K))

    for k in range(K):
        sk = s_train[y_train == labels[k], k]

        prediction_test[:,k] = conformalscore(s_test[:,k], sk)
        prediction_test[:,k] = (prediction_test[:,k] + 1) / (len(sk) + 1)
    
    return prediction_test


def train(classifier, x_train, y_train, labels, x_test, *classifier_args, **classifier_kargs):
    """ 
    Returns
    -------
    models : a list of trained models that is are used to predict
    the BCOPs score for each new observation
    """
    K = len(labels)
    models = [None]*K

    for k in range(K):
        xk = np.concatenate((x_test, x_train[y_train == labels[k],:]))
        yk = np.concatenate((np.repeat(0, np.shape(x_test)[0]),
                             np.repeat(1, np.sum(y_train == labels[k]))))
        
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
    value evaluated at a training sample using score function for atraining class.
    """
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


def evaluate_conformal(prediction, y_test, labels, alpha=0.05):
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

    for i in range(len(labels_test)):
        ii = np.where(y_test == labels_test[i])
        results[i,:] = np.apply_along_axis(np.mean, 1, prediction[ii,:] >= alpha)
    results = pd.DataFrame(results)

    results.columns = labels
    results.index = labels_test
    return results

def test():
    from sklearn.ensemble import RandomForestClassifier

    np.random.seed(123)

    X_train = np.zeros((1000, 10))
    y_train = np.zeros(1000, dtype = int)

    X_test = np.zeros((1500, 10))
    y_test = np.zeros(1500, dtype = int)

    for i in range(500):
        ## Dados para o treino
        # Classe 0
        X_train[i,:] = np.random.normal(0, 1, 10)
        y_train[i] = 0
        
        # Classe 1
        X_train[i + 500,] = np.concatenate([np.random.normal(3, 0.5, 1), np.random.normal(0, 1, 9)])
        y_train[i + 500] = 1
        
        ## Dados para o teste
        # Classe 0
        X_test[i,:] = np.random.normal(0, 1, 10)
        y_test[i] = 0
        
        # Classe 1
        X_test[i + 500,:] = np.concatenate([np.random.normal(3, 0.5, 1), np.random.normal(0, 1, 9)])
        y_test[i + 500] = 1
        
        # Classe 2 (outliers)
        X_test[i + 1000,:] = np.concatenate([np.random.normal(0, 1, 1),
                                            np.random.normal(3, 0.5, 1),
                                            np.random.normal(0, 1, 8)])
        y_test[i + 1000] = 2

    prediction_conformal = BCOPS(RandomForestClassifier, X_train, y_train, X_test)

    evaluation = evaluate_conformal(prediction_conformal, y_test, np.unique(y_train))
    print(evaluation)

if __name__ == '__main__':
    test()