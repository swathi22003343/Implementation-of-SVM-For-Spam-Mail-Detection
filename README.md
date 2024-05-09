# Implementation-of-SVM-For-Spam-Mail-Detection

## Aim:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: SWATHI D
RegisterNumber: 212222230154

import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding='windows-1252')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
### Result:
![ml_exp 9-1](https://github.com/Gopika-9266/Implementation-of-SVM-For-Spam-Mail-Detection/assets/122762773/5a262650-1000-4bc5-b9b7-7f41c28c09b8)

### data.head():
![ml_exp 9-2](https://github.com/Gopika-9266/Implementation-of-SVM-For-Spam-Mail-Detection/assets/122762773/1cbafb3d-0622-421e-beb4-d55a0de21724)

### data.info():
![ml_exp 9-3](https://github.com/Gopika-9266/Implementation-of-SVM-For-Spam-Mail-Detection/assets/122762773/0db8ec68-ec56-4521-bb03-c86716306f74)

### data.isnull().sum():
![ml_exp 9-4](https://github.com/Gopika-9266/Implementation-of-SVM-For-Spam-Mail-Detection/assets/122762773/23f7f9d8-350d-476b-8c89-e2f43d3ff7e1)

### Predicted value of y:
![ml_exp 9-5](https://github.com/Gopika-9266/Implementation-of-SVM-For-Spam-Mail-Detection/assets/122762773/303e043c-9e38-4bff-aaaa-9b55e4253650)


### Accuracy:
![ml_exp 9-6](https://github.com/Gopika-9266/Implementation-of-SVM-For-Spam-Mail-Detection/assets/122762773/e370594c-a154-410c-9b02-0a46711183b9)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

