import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel
from torch.optim import AdamW
from kobert_tokenizer import KoBERTTokenizer
from tqdm.notebook import tqdm


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

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")

# 1. 하이퍼파라미터 설정
MAX_LEN = 64
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
NUM_LABELS = len(y.unique()) # 레이블 개수 자동 인식

# 2. 토크나이저 및 데이터셋 클래스
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 데이터로더 생성
train_ds = NewsDataset(X_train, y_train, tokenizer, MAX_LEN)
test_ds = NewsDataset(X_test, y_test, tokenizer, MAX_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# 3. KoBERT 분류 모델 정의
class KoBERTClassifier(nn.Module):
    def __init__(self, num_labels):
        super(KoBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('skt/kobert-base-v1')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels) # 768 -> 레이블 개수

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits

model = KoBERTClassifier(num_labels=NUM_LABELS).to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

# 4. 학습 함수
def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# 5. 평가 함수
def eval_epoch(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# 6. 실제 학습 실행
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
    val_acc = eval_epoch(model, test_loader)
    print(f"Loss: {train_loss:.4f}, Accuracy: {val_acc*100:.2f}%")

print(f"\n>> KoBERT 최종 정확도: {val_acc * 100:.2f}%")
print("=====================================")