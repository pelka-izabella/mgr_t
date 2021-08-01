#%%
import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from text_prepper import PrepareText

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_validate

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, plot_roc_curve, roc_auc_score, matthews_corrcoef

import tensorflow as tf
#%%
def plot_confusion_matrix(y_true, y_pred):
    cf_matrix = confusion_matrix(y_true, y_pred)
    group_names = ["True Neg","False Pos","False Neg","True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                        cf_matrix.flatten()/np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    plt.show()
# %%
## Setup
in_dir = 'data'
out_dir = 'data'
raw_dataset = 'prepped.csv'

df = pd.read_csv(os.path.join(in_dir, raw_dataset), index_col='Unnamed: 0', dtype = {'rating':np.int32, 'review':np.object})

#%%
# clean
df['label'] = 0
df.loc[(df.rating == 4)|(df.rating == 5), "label"] = 1
labels = df['label']

prep = PrepareText()
df = prep.clean_text(df, in_col='review', out_col='clean_text')
df = prep.lemmatize(df, in_col='clean_text', out_col='lem', listed=False)
df.head()

#%%
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df['lem'], labels, test_size=0.2, random_state=42, stratify=labels)

# select features
# fit to train
bow = CountVectorizer(max_features=10000, lowercase=True, ngram_range=(1,1),analyzer = "word")
f = bow.fit(X_train_raw)
# tf_idf = TfidfVectorizer(max_features=10000, lowercase=True, ngram_range=(1,1),analyzer = "word", sublinear_tf=True)
# f = tf_idf.fit(X_train)

# transform test and train
X_train = f.transform(X_train_raw).toarray()
X_test = f.transform(X_test_raw).toarray()
#%%
GRID=True 

if GRID:

    model = LogisticRegression(random_state=42, n_jobs=-1, solver='saga')
    parameters = {'class_weight':('None', 'balanced'), 
                    'C':[0.1, 0.5, 1, 2, 5], 
                    'max_iter':[50,100,200],
                    'penalty':['l1', 'l2', 'none']
                    }


    # {'C': 1, 'class_weight': 'None', 'max_iter': 50}
    # {'C': 1, 'class_weight': 'balanced', 'max_iter': 100}

    # model = SVC(random_state=42)
    # parameters = {'kernel':['linear', 'rbf', 'sigmoid']}
    # {'kernel': 'rbf'}

    # model = RandomForestClassifier(random_state=42, n_jobs=-1)
    # parameters = {'n_estimators' :[50, 100,200], 'class_weight':('None', 'balanced', 'balanced_subsample')}
    # {'class_weight': 'balanced', 'n_estimators': 50}

    # model = AdaBoostClassifier(random_state=42)
    # parameters = {'learning_rate':[0.9, 0.95, 1]}
    # {'learning_rate': 0.9}

    # model = ExtraTreesClassifier(random_state=42, n_jobs=-1)
    # parameters = {'n_estimators' :[50, 100,200], 'class_weight':('None', 'balanced', 'balanced_subsample')}
    # {'class_weight': 'balanced', 'n_estimators': 100}

    scores = ['f1'] # 'precision', 'recall', 'accuracy'

    for score in scores:
        print(model)
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = RandomizedSearchCV(
            model, parameters, scoring=score
        )
        clf.fit(X_train, y_train)

        print("Best parameters set found on test set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on test set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

#%%
# selected model


# model = GaussianNB()

model = LogisticRegression(random_state=42, n_jobs=-1, penalty= 'l1', max_iter= 100, class_weight = 'None', C= 2, solver='saga')


# 0.960 (+/-0.002) for {'penalty': 'l1', 'max_iter': 200, 'class_weight': 'None', 'C': 1}
# 0.959 (+/-0.004) for {'penalty': 'none', 'max_iter': 100, 'class_weight': 'balanced', 'C': 0.1}
# 0.958 (+/-0.005) for {'penalty': 'none', 'max_iter': 50, 'class_weight': 'balanced', 'C': 1}
# 0.961 (+/-0.001) for {'penalty': 'l1', 'max_iter': 100, 'class_weight': 'None', 'C': 2}
# 0.960 (+/-0.003) for {'penalty': 'l1', 'max_iter': 50, 'class_weight': 'None', 'C': 2}
# 0.959 (+/-0.003) for {'penalty': 'none', 'max_iter': 200, 'class_weight': 'balanced', 'C': 2}
# 0.961 (+/-0.005) for {'penalty': 'none', 'max_iter': 50, 'class_weight': 'None', 'C': 0.1}
# 0.961 (+/-0.002) for {'penalty': 'none', 'max_iter': 100, 'class_weight': 'None', 'C': 1}
# 0.961 (+/-0.002) for {'penalty': 'none', 'max_iter': 100, 'class_weight': 'None', 'C': 2}
# 0.961 (+/-0.005) for {'penalty': 'none', 'max_iter': 50, 'class_weight': 'None', 'C': 2}


#%% 

model.fit(X_train, y_train)
y_true, y_pred = y_test, model.predict(X_test)
y_hat = model.predict(X_train)

# %%
print(model, "\n")
plot_confusion_matrix(y_true, y_pred)
plot_roc_curve(model, X_test, y_test)
# %%
print(model, "\n")
print("Classification report \n", classification_report(y_true,y_pred))
print("Accuracy: \t", round(accuracy_score(y_true, y_pred)*100, 2), "%")
# Precision = TruePositives / (TruePositives + FalsePositives)
print("Precision: \t", round(precision_score(y_true, y_pred)*100, 2), "%")
# Recall = TruePositives / (TruePositives + FalseNegatives)
print("Recall: \t", round(recall_score(y_true, y_pred)*100, 2), "%")
# F-Measure = (2 * Precision * Recall) / (Precision + Recall)
print("F1 score: \t", round(f1_score(y_true, y_pred)*100, 2), "%")
# MCC
print("MCC : \t", round(matthews_corrcoef(y_true, y_pred)*100, 2), "%")

# Precision: Appropriate when minimizing false positives is the focus.
# Recall: Appropriate when minimizing false negatives is the focus.
# F1-measure, which weights precision and recall equally, is the variant most often used when learning from imbalanced data.

#%%
# Accuracy
print(round(accuracy_score(y_true, y_pred)*100, 2), "%")
# Precision = TruePositives / (TruePositives + FalsePositives)
print(round(precision_score(y_true, y_pred)*100, 2), "%")
# Recall = TruePositives / (TruePositives + FalseNegatives)
print(round(recall_score(y_true, y_pred)*100, 2), "%")
# F-Measure = (2 * Precision * Recall) / (Precision + Recall)
print(round(f1_score(y_true, y_pred)*100, 2), "%")
# MCC
print(round(matthews_corrcoef(y_true, y_pred)*100, 2), "%")
# AUC
print(round(roc_auc_score(y_true, y_pred), 2))


# %%
# %%
print("Training se")
plot_confusion_matrix(y_train, y_hat)
plot_roc_curve(model, X_train, y_train)


print("Accuracy: \t", round(accuracy_score(y_train, y_hat)*100, 2), "%")
# Precision = TruePositives / (TruePositives + FalsePositives)
print("Precision: \t", round(precision_score(y_train, y_hat)*100, 2), "%")
# Recall = TruePositives / (TruePositives + FalseNegatives)
print("Recall: \t", round(recall_score(y_train, y_hat)*100, 2), "%")
# F-Measure = (2 * Precision * Recall) / (Precision + Recall)
print("F1 score: \t", round(f1_score(y_train, y_hat)*100, 2), "%")
# MCC
print("MCC : \t", round(matthews_corrcoef(y_train, y_hat)*100, 2), "%")

# Precision: Appropriate when minimizing false positives is the focus.
# Recall: Appropriate when minimizing false negatives is the focus.
# F1-measure, which weights precision and recall equally, is the variant most often used when learning from imbalanced data.
# %%
# feature importance
importances = model.coef_[0] # for LR it's coefficients and not feature imp
feat_labels = bow.get_feature_names()
indices = np.argsort(importances)[::-1]

# worst
for f in range(len(feat_labels)-1, len(feat_labels)-11, -1):
    print(feat_labels[indices[f]], "\t", importances[indices[f]])
    #print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
print("_"*30)
# best
for f in range(10):
    print(feat_labels[indices[f]], "\t", importances[indices[f]])
    #print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
#%%
plt.figure(figsize=(10,5))
plt.title('Importances of features')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()
# %%
X_test[(y_test == 1) & (y_pred.T == 0)]
# %%
