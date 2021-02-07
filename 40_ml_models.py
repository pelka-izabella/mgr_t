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
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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

df['label'] = 0
df.loc[(df.rating == 4)|(df.rating == 5), "label"] = 1
labels = df['label']

prep = PrepareText()
df = prep.clean_text(df, in_col='review', out_col='clean_text')
df = prep.lemmatize(df, in_col='clean_text', out_col='lem', listed=False)
df.head()
#%%
# select features
bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
features = bow.fit_transform(df['lem']).toarray()

#%%
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
#%%
GRID=True 

if GRID:

    model = LogisticRegression(random_state=42, n_jobs=-1)
    parameters = {'class_weight':('None', 'balanced'), 'C':[1, 2, 5], 'max_iter':[50,100,200]}
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

    scores = ['f1'] # 'precision', 'recall'

    for score in scores:
        print(model)
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = RandomizedSearchCV(
            model, parameters, scoring='%s_macro' % score
        )
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
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
 
estimators = [
    ('svc', SVC(kernel='rbf')),  
    ('lr', LogisticRegression(random_state=42, n_jobs=-1, C=1, class_weight='balanced', max_iter=100))
]

final_estimator = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=50, class_weight='balanced')

model = StackingClassifier(cv=3, estimators=estimators, final_estimator=final_estimator)

# model = LogisticRegression(random_state=42, n_jobs=-1, C=1, class_weight='balanced', max_iter=100)

# model = SVC(random_state=42, kernel='rbf')

# model = RandomForestClassifier(random_state=42, n_jobs=-1,class_weight='balanced',n_estimators=100)

# model = AdaBoostClassifier(random_state=42, learning_rate=0.9)

#  model = ExtraTreesClassifier(random_state=42, n_jobs=-1, class_weight='balanced',n_estimators=100)

#%% 

model.fit(X_train, y_train)
y_true, y_pred = y_test, model.predict(X_test)

# %%
print(model, "\n")
plot_confusion_matrix(y_true, y_pred)
# %%
print(model, "\n")
print("Classification report \n", classification_report(y_true,y_pred))
print("Accuracy: \t", round(accuracy_score(y_true, y_pred)*100, 4), "%")
# Precision = TruePositives / (TruePositives + FalsePositives)
print("Precision: \t", round(precision_score(y_true, y_pred)*100, 4), "%")
# Recall = TruePositives / (TruePositives + FalseNegatives)
print("Recall: \t", round(recall_score(y_true, y_pred)*100, 4), "%")
# F-Measure = (2 * Precision * Recall) / (Precision + Recall)
print("F1 score: \t", round(f1_score(y_true, y_pred)*100, 4), "%")

# Precision: Appropriate when minimizing false positives is the focus.
# Recall: Appropriate when minimizing false negatives is the focus.
# F1-measure, which weights precision and recall equally, is the variant most often used when learning from imbalanced data.
# %%
