from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, r2_score

def test_classifier(which_classifier, x_train, y_train, x_test, y_test):
        classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=10000), 
        'K Neighbors': KNeighborsClassifier(p=2, metric='minkowski', n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(criterion='entropy', random_state=0),
        'Random Forest': RandomForestClassifier(criterion='entropy', n_estimators=300, random_state=0),
        'Linear SVM': SVC(kernel='linear', random_state=0),
        'RBF SVM': SVC(kernel='rbf', random_state=0),
        'Naive Bayes': GaussianNB()
        }

        try:
                classifier = classifiers[which_classifier]
        except KeyError:
                print('Sorry, that classifier is not valid.')
        else:
                classifier.fit(x_train, y_train)
                predictions = classifier.predict(x_test)

                cm = confusion_matrix(y_test, predictions)
                accuracy = accuracy_score(y_test, predictions)
                f1 = f1_score(y_test, predictions)

                print('Confusion Matrix:')
                print(cm)
                print(f'Accuracy: {accuracy}')
                print(f'F1 Score: {f1}')