# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# 데이터 로드
train_data = pd.read_csv('titanic/train.csv')
test_data = pd.read_csv('titanic/test.csv')

# 데이터 전처리
def preprocess_data(data):
    # 결측치 처리
    data['Age'] = data.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    
    # 새로운 피처 생성
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    data['FarePerPerson'] = data['Fare'] / data['FamilySize']
    
    # 범주형 변수 인코딩
    data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)
    
    return data

# 데이터 전처리 적용
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# 사용할 피처 선택
features = ['Pclass', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'FarePerPerson', 'Sex_male', 'Embarked_Q', 'Embarked_S']
X = train_data[features]
y = train_data['Survived']
X_test = test_data[features]

# 학습/검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습 (XGBoost 사용)
model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)

# 검증 데이터로 예측 및 정확도 평가
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"검증 데이터 정확도: {accuracy:.4f}")

# 하이퍼파라미터 튜닝 (GridSearchCV)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}
grid_search = GridSearchCV(estimator=XGBClassifier(random_state=42),
                           param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# 최적 모델로 재학습
best_model = grid_search.best_estimator_
best_model.fit(X, y)

# 테스트 데이터 예측
test_predictions = best_model.predict(X_test)

# 제출 파일 생성
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': test_predictions})
submission.to_csv('submission.csv', index=False)
print("submission.csv 파일이 생성되었습니다.")