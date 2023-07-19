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

def train(classifier, x, y, labels, xte, *classifier_args, **classifier_kargs):
    K = len(labels)
    models = [None]*K

    for k in range(K):
        xk = np.concatenate((xte, x[y == labels[k],:]))
        yk = np.concatenate((np.repeat(0, np.shape(xte)[0]),
                             np.repeat(1, np.sum(y == labels[k]))))
        
        models[k] = classifier(*classifier_args, **classifier_kargs).fit(xk, yk)

    return models

def prediction(models, x, y, labels, xte):
    K = len(labels)
    s = np.zeros((len(y), K))
    ste = np.zeros((np.shape(xte)[0], K))

    for k in range(K):    
        if np.sum(y == labels[k]) > 0:
            temp1 = models[k].predict_proba(x)
            temp2 = models[k].predict_proba(xte)

            try:
                temp3 = np.shape(temp1)[1]
            except IndexError:
                s[:,k] = temp1
                ste[:,k] = temp2
            else:
                s[:,k] = temp1[:,temp3-1]
                ste[:,k] = temp2[:,temp3-1]

    prediction_conformal = conformal_scores(ste = ste, s = s, y = y, labels = labels)
    return {'prediction_conformal':prediction_conformal, 'score_test':ste, 'score_train':s}


def BCOPS(classifier, x1, y1, xte1, x2, y2, xte2, labels, *classifier_args, **classifier_kargs):
    # Training
    models1 = train(classifier, x2, y2, labels, xte2, *classifier_args, **classifier_kargs)
    models2 = train(classifier, x1, y1, labels, xte1, *classifier_args, **classifier_kargs)
    
    # Prediction (melhorar os argumentos posicionais)
    prediction1 = prediction(models1, x1, y1, labels, xte1)['prediction_conformal']
    prediction2 = prediction(models2, x2, y2, labels, xte2)['prediction_conformal']
    return {'conformal_scores1':prediction1, 'conformal_scores2':prediction2}

def evaluate_conformal(prediction, yte, labels, alpha=0.05):
    labels_te = np.unique(yte)
    res = np.zeros((len(labels_te), len(labels)))

    for i in range(len(labels_te)):
        ii = np.where(yte == labels_te[i])
        res[i,:] = np.apply_along_axis(np.mean, 1, prediction[ii,:] >= alpha)
    res = pd.DataFrame(res)

    res.columns = labels
    res.index = labels_te
    return res

def main():
    from sklearn.ensemble import RandomForestClassifier

    np.random.seed(123)

    X_train = np.zeros((1000, 10))
    y_train = np.zeros(1000)

    X_test = np.zeros((1500, 10))
    y_test = np.zeros(1500)

    for i in range(500):
        ## Dados para o treino
        # Classe 1
        X_train[i,:] = np.random.normal(0, 1, 10)
        y_train[i] = 0
        
        # Classe 2
        X_train[i + 500,] = np.concatenate([np.random.normal(3, 0.5, 1), np.random.normal(0, 1, 9)])
        y_train[i + 500] = 1
        
        ## Dados para o teste
        # Classe 1
        X_test[i,:] = np.random.normal(0, 1, 10)
        y_test[i] = 0
        
        # Classe 2
        X_test[i + 500,:] = np.concatenate([np.random.normal(3, 0.5, 1), np.random.normal(0, 1, 9)])
        y_test[i + 500] = 1
        
        # Classe 3 (outliers)
        X_test[i + 1000,:] = np.concatenate([np.random.normal(0, 1, 1),
                                            np.random.normal(3, 0.5, 1),
                                            np.random.normal(0, 1, 8)])
        y_test[i + 1000] = 2

    foldid = np.random.randint(1, 3, len(y_train))
    foldid_te = np.random.randint(1, 3, len(y_test))

    xtrain1 = X_train[foldid == 1,:]
    xtrain2 = X_train[foldid == 2,:]
    ytrain1 = y_train[foldid == 1]
    ytrain2 = y_train[foldid == 2]
    xtest1 = X_test[foldid_te == 1,:]
    xtest2 = X_test[foldid_te == 2,:]
    labels = np.unique(y_train)

    bcops = BCOPS(RandomForestClassifier, xtrain1, ytrain1, xtest1, xtrain2, ytrain2, xtest2, labels)
    
    prediction_conformal = np.zeros((len(y_test), len(labels)))
    prediction_conformal[foldid_te==1,:] = bcops['conformal_scores1']
    prediction_conformal[foldid_te==2,:] = bcops['conformal_scores2']

    print(evaluate_conformal(prediction_conformal, y_test, labels))

if __name__ == '__main__':
    main()
