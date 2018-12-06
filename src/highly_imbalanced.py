import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import warnings


class HighlyImbalanced():
    print()


def cross_validate(X, y, k, boost):
    scores = np.zeros(k)
    for i in range(k):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=1/k, random_state=i)
        model, f1 = boost(X_train, X_test, y_train, y_test)
        scores[i] = f1
    return np.average(scores)


def xgb(X_train, X_test, y_train, y_test):
    # fit model no training data
    model = XGBClassifier(eta=0.15, max_depth=8, n_estimators=200,
                            random_state=42, max_delta_step=50, min_child_weight = 2)
    model.fit(X_train, y_train)
    print("================================================================")
    print(model)
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    f1 = f1_score(y_test, y_pred, average='binary')
    print("F1 Score: %.2f%%" % (f1 * 100.0))
    print("================================================================")
    return model, f1


def smote_xgb(X_train, X_test, y_train, y_test):
    # SMOTE
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)
    # fit model no training data
    model = XGBClassifier(eta=0.15, max_depth=8, n_estimators=200,
                            random_state=42, max_delta_step=50, min_child_weight = 2)
    model.fit(X_train_sm, y_train_sm)
    print("================================================================")
    print(model)
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    f1 = f1_score(y_test, y_pred, average='binary')
    print("F1 Score: %.2f%%" % (f1 * 100.0))
    print("================================================================")
    return model, f1


def main():
    X = pd.read_csv(os.path.join(os.getcwd()[:-3],
                                  'bin/train.csv'), low_memory=False, header=None).as_matrix()
    y = pd.read_csv(os.path.join(os.getcwd()[:-3],
                                  'bin/train_labels.csv'),
                                    low_memory=False, header=None).as_matrix()
    print("Dataset: X{}, y{}".format(X.shape, y.shape))

    xgb_score = cross_validate(X, y, 3, xgb)
    print(xgb_score)

    smote_score = cross_validate(X, y, 3, smote_xgb)
    print(smote_score)






if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()