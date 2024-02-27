from neucube import Reservoir
from neucube.validation import Pipeline
from neucube.sampler import SpikeCount

from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

from tqdm import tqdm


def get_classifier(clf_type: str = "regression"):
    if clf_type == "regression":
        return LogisticRegression(solver='liblinear')
    if clf_type == "random_forest":
        return RandomForestClassifier()
    if clf_type == "xgboost":
        return XGBClassifier()
    if clf_type == "naive_bayes":
        return MultinomialNB()
    return SVC(kernel='linear')


def snn_experiment(data_x, data_y, clf_type: str = "regression", seed: int = 123):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    y_total, pred_total = [], []

    for train_index, test_index in tqdm(kf.split(data_x)):
        x_train, x_test = data_x[train_index], data_x[test_index]
        y_train, y_test = data_y[train_index], data_y[test_index]

        res = Reservoir(inputs=data_x.shape[2], cube_shape=(8, 8, 8))
        sam = SpikeCount()
        clf = get_classifier(clf_type)
        pipe = Pipeline(res, sam, clf)

        pipe.fit(x_train, y_train, train=True)
        pred = pipe.predict(x_test)

        y_total.extend(y_test)
        pred_total.extend(pred)
    print(accuracy(y_total, pred_total))
    print(confusion_matrix(y_total, pred_total))


def lsa_experiment(data_x, data_y, clf_type: str = "regression", seed: int = 123):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    y_total, pred_total = [], []

    for train_index, test_index in tqdm(kf.split(data_x)):
        x_train, x_test = data_x[train_index], data_x[test_index]
        y_train, y_test = data_y[train_index], data_y[test_index]

        clf = get_classifier(clf_type)
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)

        y_total.extend(y_test)
        pred_total.extend(pred)
    print(accuracy(y_total, pred_total))
    print(confusion_matrix(y_total, pred_total))
