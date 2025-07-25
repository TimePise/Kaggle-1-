# 🚢 Titanic 생존자 예측 (Kaggle)

이 프로젝트는 [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) 대회의 데이터셋을 기반으로, 탑승자의 정보를 바탕으로 생존 여부를 예측하는 머신러닝 모델을 구축한 것입니다.

---

## 🛠 사용 기술

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost (분류 모델)
- GridSearchCV (하이퍼파라미터 튜닝)

---

## ✨ 주요 기능

- **결측치 처리**: Age, Fare, Embarked 컬럼 처리
- **파생 변수 생성**: FamilySize, IsAlone, FarePerPerson
- **범주형 변수 인코딩**: Sex, Embarked
- **XGBoost 모델 적용**
- **GridSearchCV**를 활용한 최적 하이퍼파라미터 탐색
- **submission.csv** 생성

---

## 📊 성능 평가

- 학습 데이터를 80:20으로 분리하여 **검증 정확도(accuracy)**를 출력
- 최적 하이퍼파라미터 모델로 전체 학습 데이터 재학습 후 테스트 예측

---

## ▶️ 실행 방법

```bash
# 1. 필요한 라이브러리 설치 (예: pip install -r requirements.txt)
# 2. 파일 실행
python titanic.py
