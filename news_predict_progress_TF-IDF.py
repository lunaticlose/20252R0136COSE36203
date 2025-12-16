import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 데이터 로드
try:
    df = pd.read_csv('train_data.csv', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('train_data.csv', encoding='euc-kr')

print(f"데이터 로드 완료: {len(df)}개 샘플")

# 2. 데이터 전처리 (결측치 제거)
df = df.dropna(subset=['title']) # 'title' 컬럼의 NaN 값 제거
df = df.dropna(subset=['topic_idx']) # 'topic_idx' 컬럼의 NaN 값 제거

# 3. 데이터 분리
X = df['title'] # 뉴스 기사 제목
y = df['topic_idx'] # 뉴스 주제

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TF-IDF 벡터화
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. 모델 학습 및 평가
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("--- 베이스라인(TF-IDF + 로지스틱 회귀) 결과 ---")
print(f'정확도: {accuracy * 100:.2f}%')