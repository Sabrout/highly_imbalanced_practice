#Import libraries:
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks


class Imbalanced():
    print()


def main():
    # Reading Dataset
    X = pd.read_csv(os.path.join(os.getcwd()[:-3],
                                  'bin/train.csv'), low_memory=False, header=None).as_matrix()
    y = pd.read_csv(os.path.join(os.getcwd()[:-3],
                                  'bin/train_labels.csv'),
                                    low_memory=False, header=None).as_matrix()
    # Splitting Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2, random_state=42)

    print("__________________________________________________________________")

    # Gradient Boosting
    model = GradientBoostingClassifier(min_samples_split=100, min_samples_leaf=5,
                                       max_depth=10, max_features='sqrt')
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_test, y_test, cv=5, scoring='f1_macro')
    print("F1 Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(model, X_test, y_test, cv=5, scoring='precision')
    print("Precision: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(model, X_test, y_test, cv=5, scoring='recall')
    print("Recall: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print("__________________________________________________________________")

    # Gradient Boosting with SMOTE
    sm = SMOTE(random_state=42)
    y = pd.read_csv(os.path.join(os.getcwd()[:-3],
                                  'bin/train_labels.csv'),
                                    low_memory=False, header=None).as_matrix()
    X_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)
    model2 = GradientBoostingClassifier()
    model2.fit(X_train_sm, y_train_sm)
    scores = cross_val_score(model2, X_test, y_test, cv=5, scoring='f1_macro')
    print("F1 Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(model2, X_test, y_test, cv=5, scoring='precision')
    print("Precision: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(model2, X_test, y_test, cv=5, scoring='recall')
    print("Recall: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print("__________________________________________________________________")

    # Gradient Boosting with Tomek Links
    tl = TomekLinks(random_state=42)
    X_train_tl, y_train_tl = tl.fit_sample(X_train, y_train)
    model3 = GradientBoostingClassifier()
    model3.fit(X_train_tl, y_train_tl)
    scores = cross_val_score(model3, X_test, y_test, cv=5, scoring='f1_macro')
    print("F1 Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(model3, X_test, y_test, cv=5, scoring='precision')
    print("Precision: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(model3, X_test, y_test, cv=5, scoring='recall')
    print("Recall: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print("__________________________________________________________________")

    # Test Dataset
    sm = SMOTE(random_state=42)
    X_sm, y_sm = sm.fit_sample(X, y)
    model4 = GradientBoostingClassifier(min_samples_split=100, min_samples_leaf=5,
                                       max_depth=10, max_features='sqrt')
    model4.fit(X_sm, y_sm)
    X_test = pd.read_csv(os.path.join(os.getcwd()[:-3],
                                  'bin/test.csv'), low_memory=False, header=None).as_matrix()
    y_pred = model4.predict(X_test)
    predictions = [round(value) for value in y_pred]
    pd.DataFrame(predictions, columns=['predictions']).to_csv('prediction.csv')

if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()