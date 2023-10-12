import timeit
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import time
from IPython.core.display_functions import display
from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import dtreeviz
import datetime
import warnings
import random


random.seed(25)

warnings.filterwarnings("ignore", category=FutureWarning)

max_depth = 5
min_samples_split = 10
min_samples_leaf = 5
max_features = "sqrt" #per limitare il numero di feature considerate in ogni suddivisione
random_state = 25


# Dizionario dei classificatori da utilizzare
dict_classifiers = {
    "Naive Bayes": GaussianNB(),
    "Support Vector Classification": SVC(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            max_features=max_features,
                                            random_state=random_state),
    "Random Forest": RandomForestClassifier(n_estimators=5),
    "Extra Trees": ExtraTreesClassifier(max_depth=6,
                                        min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf,
                                        max_features=max_features,
                                        random_state=random_state,
                                        n_estimators=10
                                        ),
    "AdaBoost": AdaBoostClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=10),
    "XGBClassifier": XGBClassifier(),
}


def batch_classify(X_train, y_train, X_test, y_test, no_classifiers=5, verbose=False):
    """
    Questo metodo prende in input le matrici X e Y del set di addestramento e del set di test.
    Addestra tutti i classificatori specificati nel dizionario dict_classifiers.
    I modelli addestrati e le relative accuratezze vengono salvati in un dizionario per l'analisi.
    """
    df_= pd.read_csv("data.csv", sep=",")
    del df_['name']
    feature_names = df_.drop(columns=['status']).columns.tolist()
    class_names = ['0', '1']
    dict_models = {}
    for classifier_name, classifier in list(dict_classifiers.items())[:no_classifiers]:
        t_start = timeit.default_timer()
        classifier.fit(X_train, y_train)
        t_end = timeit.default_timer()
        t_diff = (t_end - t_start)

        y_pred = classifier.predict(X_test)
        ac = metrics.accuracy_score(y_test, y_pred)
        pr, rc, fs, sup = metrics.precision_recall_fscore_support(y_test, y_pred, average='macro')
        accuracy_score = round(ac, 3)
        precision_score = round(pr, 3)
        recall_score = round(rc, 3)
        f_score = round(fs, 3)

        if 'XGBClassifier' in classifier_name:
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.show()



        dict_models[classifier_name] = {'model': classifier, 'Accuracy': accuracy_score, 'train_time': t_diff,
                                        'Precision': precision_score, 'Recall': recall_score,
                                        'F-Score': f_score}
        if verbose:
            print("Addestramento {c} in {f:.2f} s".format(c=classifier_name, f=t_diff))
            print(classification_report(y_test, y_pred))

    return dict_models


# Funzione per visualizzare i risultati dei modelli
def display_dict_models(dict_models, sort_by='Accuracy'):
    cls = [key for key in dict_models.keys()]
    accuracy_s = [dict_models[key]['Accuracy'] for key in cls]
    training_t = [dict_models[key]['train_time'] for key in cls]
    precision_s = [dict_models[key]['Precision'] for key in cls]
    recall_s = [dict_models[key]['Recall'] for key in cls]
    f_s = [dict_models[key]['F-Score'] for key in cls]

    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls), 6)),
                       columns=['classifier', 'Accuracy', 'Precision', 'Recall',
                                'F-Score', 'train_time'])
    for ii in range(0, len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'Accuracy'] = accuracy_s[ii]
        df_.loc[ii, 'Precision'] = precision_s[ii]
        df_.loc[ii, 'Recall'] = recall_s[ii]
        df_.loc[ii, 'F-Score'] = f_s[ii]
        df_.loc[ii, 'train_time'] = training_t[ii]

    display(df_)
    return df_
