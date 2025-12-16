import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel
from torch.optim import AdamW
from kobert_tokenizer import KoBERTTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm.notebook import tqdm # 진행률 표시

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



print("\n===== 데이터 로드 및 전처리 =====")
# Colab에 업로드한 'train_data.csv' 파일 읽기
try:
    df = pd.read_csv('train_data.csv', encoding='utf-8')
except FileNotFoundError:
    print("\n*** 에러: 'train_data.csv' 파일을 Colab에 업로드했는지 확인하세요. ***\n")
    raise
except UnicodeDecodeError:
    df = pd.read_csv('train_data.csv', encoding='euc-kr')

# 결측치 제거
df = df.dropna(subset=['title', 'topic_idx'])

df = df.sample(n=10000, random_state=42)
print(f"로드된 샘플 수: {len(df)}")

# 데이터셋 분리 (Train: 80%, Test: 20%)
X = df['title']
y = df['topic_idx']
num_labels = len(y.unique()) # 7 (DACON 기준 7개 주제)
print(f"총 레이블 개수: {num_labels}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train 샘플: {len(X_train)}, Test 샘플: {len(X_test)}")


print("\n===== 토크나이저 및 데이터셋 정의 =====")
# KoBERT 토크나이저 로드
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

# PyTorch 커스텀 데이터셋 클래스
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]

        # KoBERT 토크나이저로 텍스트 인코딩
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,    # [CLS], [SEP] 추가
            max_length=self.max_len,    # 최대 길이 64
            return_token_type_ids=False,
            padding='max_length',       # 패딩
            truncation=True,            # 잘라내기
            return_attention_mask=True, # 어텐션 마스크 반환
            return_tensors='pt',        # PyTorch 텐서로 반환
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 데이터셋 및 데이터로더 생성
train_dataset = NewsDataset(X_train, y_train, tokenizer)
test_dataset = NewsDataset(X_test, y_test, tokenizer)

BATCH_SIZE = 32 # 배치 사이즈
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("데이터로더 생성 완료.")


print("\n===== KoBERT 분류 모델 정의 =====")
class KoBERTClassifier(nn.Module):
    def __init__(self, bert, num_labels):
        super(KoBERTClassifier, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(p=0.1)
        # BERT 출력(768) -> 분류할 레이블 개수(num_labels)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        # BERT 모델 실행
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # [CLS] 토큰에 해당하는 pooler_output 사용
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# 모델 로드
bert_model = BertModel.from_pretrained('skt/kobert-base-v1')
model = KoBERTClassifier(bert_model, num_labels=num_labels).to(device)
print("모델 로드 완료.")



print("\n===== 학습 및 평가 루프 정의 =====")
# 옵티마이저 (AdamW) 및 손실 함수 (CrossEntropyLoss)
optimizer = AdamW(model.parameters(), lr=2e-5) # 2e-5가 BERT Fine-tuning에 권장됨
loss_fn = nn.CrossEntropyLoss().to(device)

# --- 학습 함수 ---
def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train() # 학습 모드
    total_loss = 0

    for batch in tqdm(data_loader, desc="[Train]"):
        # 데이터를 GPU로 이동
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 순전파
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()

        # 역전파
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(data_loader)

# --- 평가 함수 ---
def eval_model(model, data_loader, loss_fn, device):
    model.eval() # 평가 모드
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad(): # 그래디언트 계산 비활성화
        for batch in tqdm(data_loader, desc="[Eval]"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, dim=1) # 가장 높은 확률의 인덱스

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(data_loader)
    return accuracy, avg_loss

print("학습/평가 함수 정의 완료.")



print("\n===== 모델 학습 시작 =====")
EPOCHS = 3 
final_val_acc = 0.0

for epoch in range(EPOCHS):
    print(f'--- Epoch {epoch + 1}/{EPOCHS} ---')
    train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
    print(f'Train Loss: {train_loss:.4f}')

    val_acc, val_loss = eval_model(model, test_loader, loss_fn, device)
    print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc * 100:.2f}%')
    final_val_acc = val_acc # 마지막 에포크의 정확도를 저장

print("===== 학습 완료 =====")

print(f"*** 최종 테스트 정확도: {final_val_acc * 100:.2f}% ***")
