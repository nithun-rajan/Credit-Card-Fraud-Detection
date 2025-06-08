import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix , accuracy_score , classification_report

credit_card_data = pd.read_csv('creditcard.csv')

credit_card_data['Class'].value_counts()
#print(credit_card_data['Class'].value_counts())

normal = credit_card_data[credit_card_data.Class==0]
fraud = credit_card_data[credit_card_data.Class==1]

#print(normal.shape)
#print(fraud.shape)

labels = ["Normal", "Fraud"]
count_classes = credit_card_data.value_counts(credit_card_data['Class'], sort=True)
print(count_classes)
count_classes.plot(kind="bar", rot=0)

plt.ylabel("Count")

plt.xticks(range(2), labels)

plt.show()

normal.Amount.describe()
fraud.Amount.describe()

sns.relplot(x='Amount', y= 'Time', hue='Class', data=credit_card_data)

plt.show()

normal_sample = normal.sample(n=492)

credit_card_new_data=pd.concat([normal_sample,fraud], axis=0)

X = credit_card_new_data.drop('Class', axis=1)
Y = credit_card_new_data['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
print(X.shape, X_train.shape, X_test.shape)

# Creating Model:
model = LogisticRegression()
# training the Logistic Regression model with training data:
model.fit(X_train,Y_train)

X_train_pred = model.predict(X_train)

training_data_accuracy = accuracy_score(X_train_pred, Y_train)
print('Accuracy of Training data:', training_data_accuracy)


X_test_predict = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_predict,Y_test)

print('Accuracy of Testing data:', test_data_accuracy)

print(confusion_matrix(X_test_predict, Y_test))
print(classification_report(X_test_predict, Y_test))


