import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


try:
    df = pd.read_csv('train_data.csv', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('train_data.csv', encoding='euc-kr')

# 데이터 전처리 (결측치 제거)
df = df.dropna(subset=['title', 'topic_idx'])


df = df.sample(n=10000, random_state=42)

# 입력(X)과 정답(y) 분리
X = df['title']      # 뉴스 제목
y = df['topic_idx']  # 정답 레이블 (가짜뉴스: 0/1, 뉴스토픽: 0~6)

# 학습용/테스트용 데이터 분리 (8:2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"데이터 준비 완료! 학습 데이터: {len(X_train)}개, 테스트 데이터: {len(X_test)}개")


print("=== 베이스라인 모델 학습 시작 ===")

# 1. TF-IDF 벡터화 (텍스트 -> 숫자 변환)
vectorizer = TfidfVectorizer(max_features=5000) # 상위 5000개 단어만 사용
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 2. 로지스틱 회귀 모델 학습
baseline_model = LogisticRegression(max_iter=1000, n_jobs=-1)
baseline_model.fit(X_train_vec, y_train)

# 3. 예측 및 평가
y_pred_base = baseline_model.predict(X_test_vec)
acc_base = accuracy_score(y_test, y_pred_base)

print(f"\n>> 베이스라인(TF-IDF) 정확도: {acc_base * 100:.2f}%")
print("=====================================")