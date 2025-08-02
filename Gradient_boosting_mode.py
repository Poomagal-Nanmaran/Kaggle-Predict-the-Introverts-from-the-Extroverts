import pandas as pd

# Load datasets (adjust paths if needed)
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(train.shape, test.shape)
print(train.head())

#----------------------------Exploratory Data Analysis ---------------------
print(train.info())
print(train['Personality'].value_counts())
print(train.isnull().sum())  # check missing values


#-------------- preprocessing ------------------
from sklearn.preprocessing import LabelEncoder

# Encode target
le = LabelEncoder()
train['Personality'] = le.fit_transform(train['Personality'])  # Extrovert=0, Introvert=1

# Combine train & test for consistent preprocessing
test_ids = test['id']
X = pd.concat([train.drop(columns=['Personality']), test], axis=0)

# Encode Yes/No
for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].map({'Yes':1, 'No':0})

# Fill missing
X = X.fillna(X.median())

# Split back
X_train = X.iloc[:len(train)]
X_test = X.iloc[len(train):]
y_train = train['Personality']

#--------------------- Model training --------------------
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

# Create model
lgbm = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    random_state=42
)

# Cross-validation
scores = cross_val_score(lgbm, X_train, y_train, cv=5, scoring='accuracy')
print(f"LightGBM CV Accuracy: {scores.mean():.4f}")

# Fit full model
lgbm.fit(X_train, y_train)

#--------------------- Predictions --------------------
preds = lgbm.predict(X_test)
preds_labels = le.inverse_transform(preds)

submission = pd.DataFrame({
    'id': test_ids,
    'Personality': preds_labels
})
submission.to_csv("submission_lgbm.csv", index=False)
print("Saved submission_lgbm.csv")

