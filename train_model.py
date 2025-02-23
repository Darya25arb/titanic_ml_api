import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle



url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

df["Age"] = df["Age"].fillna(df["Age"].median())
df.drop(columns=["Cabin"], inplace=True)
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])



# выбираем нужные столбцы
df_model = df[["Survived", "Pclass", "Sex", "Age", "Fare", "Embarked"]].copy()


# кодируем категориальные переменные
df_model["Sex"] = LabelEncoder().fit_transform(df_model["Sex"])
df_model["Embarked"] = LabelEncoder().fit_transform(df_model["Embarked"])


# разделяем на признаки (Х) и целевую переменную (у)
X = df_model.drop(columns=["Survived"])
y = df_model["Survived"]


# разделяем на train и test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# создаем и обучаем модель случайного леса
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


with open('model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)


print("Модель обучена и сохранена в model.pkl")