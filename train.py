import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score,accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import skops.io as sio

df=pd.read_csv(r"Data/drug.csv")
df=df.sample(frac=1)
df.head()

X=df[["Age","Sex","BP","Cholesterol","Na_to_K"]]
y=df["Drug"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

cat_col=[1,2,3]
num_col=[0,4]

transform =ColumnTransformer(
    [
        ("encoder",OrdinalEncoder(),cat_col),
        ("num_imputer",SimpleImputer(strategy="median"),num_col),
        ("num_scaler",StandardScaler(),num_col),
    ]
)

pipe=Pipeline(
    steps=[
        ("preprocessing",transform),
        ("model",RandomForestClassifier(n_estimators=100,random_state=42)),
    ]
)

pipe.fit(X_train,y_train)

predictions=pipe.predict(X_test)
accuracy=accuracy_score(y_test,predictions)
f1=f1_score(y_test,predictions,average="macro")

print(f"Accuracy {accuracy*100}% | F1 {f1}")

cm=confusion_matrix(y_test,predictions,labels=pipe.classes_)
plt.figure(figsize=(12,8))
Con=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=
                           pipe.classes_)
Con.plot()

plt.savefig("Results/img.png",dpi=120)

with open("Results/metrics.txt","w") as f:
    f.write(f"Accuracy: {accuracy*100}% | F1: {f1*100}")

sio.dump(pipe,"Model/git.skops")
