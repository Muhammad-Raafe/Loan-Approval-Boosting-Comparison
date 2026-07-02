import pandas as pd
from sklearn.model_selection import train_test_split    
from sklearn.metrics import (accuracy_score,confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier


df=pd.read_csv('data.csv')
print(df.info())

print(df.isnull().sum())   



print(df["Married"].value_counts())
print(df["Loan_Status"].value_counts())
print(df["Property_Area"].value_counts())
print(df["Self_Employed"].value_counts())
print(df["Dependents"].value_counts())
print(df["Loan_Amount_Term"].value_counts())
print(df["Credit_History"].value_counts())


df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
df["Married"].fillna(df["Married"].mode()[0], inplace=True)
df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)
df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median(), inplace=True)
df["Credit_History"].fillna(df["Credit_History"].median(), inplace=True)




le=LabelEncoder()
df["Married"]=le.fit_transform(df["Married"])
df["Gender"]=le.fit_transform(df["Gender"])
df["Loan_Status"]=le.fit_transform(df["Loan_Status"])
df["Self_Employed"]=le.fit_transform(df["Self_Employed"])


df=pd.get_dummies(df, columns=["Dependents","Education","Property_Area"], drop_first=True)


x=df.drop(columns=["Loan_Status","Loan_ID"])
y=df["Loan_Status"]

print(x.dtypes)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42,stratify=y)

cat_model=CatBoostClassifier(random_state=42,verbose=0)

cat_param_grid={
    'iterations':[100,200,300],
    'depth':[3,7,5],
    'learning_rate':[0.01,0.1,0.2],
    'l2_leaf_reg':[1,3,5],
    'random_strength':[0.5,1.0,1.5]
}

cat_grid=GridSearchCV(estimator=cat_model, param_grid=cat_param_grid, cv=5, scoring='accuracy', n_jobs=-1)

cat_grid.fit(x_train,y_train)
cat_prediction=cat_grid.predict(x_test)

print("Best Parameters:", cat_grid.best_params_)
print("Accuracy:", accuracy_score(y_test, cat_prediction))
print("Confusion Matrix:\n", confusion_matrix(y_test, cat_prediction))
print("Classification Report:\n", classification_report(y_test, cat_prediction))






model=xgb.XGBClassifier(random_state=42,eval_metric='logloss')

xg_param_grid={
    'n_estimators':[100,200,300],
    'max_depth':[3,7,5],
    'learning_rate':[0.01,0.1,0.2],
    'subsample':[0.8,0.9,1.0],
    'colsample_bytree':[0.8,0.9,1.0],
    "gamma":[0,0.1,0.2],
}

xg_grid=GridSearchCV(estimator=model, param_grid=xg_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
xg_grid.fit(x_train, y_train)
xg_prediction=xg_grid.predict(x_test)

print("Best Parameters:", xg_grid.best_params_)
print("Accuracy:", accuracy_score(y_test, xg_prediction))
print("Confusion Matrix:\n", confusion_matrix(y_test, xg_prediction))
print("Classification Report:\n", classification_report(y_test, xg_prediction))





gb_model=LGBMClassifier(random_state=42)

gb_param_grid={
    'n_estimators':[100,200,300],
    'max_depth':[3,7,5],
    'learning_rate':[0.01,0.1,0.2],
    'subsample':[0.8,0.9,1.0],
    'colsample_bytree':[0.8,0.9,1.0],
    'num_leaves':[31,50,100],


}

gb_grid=GridSearchCV(estimator=gb_model, param_grid=gb_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gb_grid.fit(x_train, y_train)
gb_prediction=gb_grid.predict(x_test)

print("Best Parameters:", gb_grid.best_params_)
print("Accuracy:", accuracy_score(y_test, gb_prediction))
print("Confusion Matrix:\n", confusion_matrix(y_test, gb_prediction))
print("Classification Report:\n", classification_report(y_test, gb_prediction))




comparison_df = pd.DataFrame({
    'Model': ['CatBoost', 'XGBoost', 'LightGBM'],
    'Accuracy': [accuracy_score(y_test, cat_prediction), accuracy_score(y_test, xg_prediction), accuracy_score(y_test, gb_prediction)],
})

print(comparison_df)

sns.barplot(x='Model', y='Accuracy', data=comparison_df)

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.show()
