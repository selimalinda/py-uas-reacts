from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

#import dataset
x,y = load_iris(return_X_y=True)

print(f'Dimensi Feature :{x.shape}')
print(f'Class : {set(y)}')

#splite dataset
x_train,x_test,y_train,y_test = train_test_split(x,
                                                 y,
                                                 test_size=0.3,
                                                 random_state=0)
#classification
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(x_train,y_train)

#evaluasi model
y_pred = model.predict(x_test)
print(classification_report(y_test,y_pred))