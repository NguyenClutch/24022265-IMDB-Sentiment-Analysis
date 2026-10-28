"""Run the full IMDB experiment. Outputs are written to results/."""
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)
# Bỏ dấu # ở dòng dưới nếu môi trường chưa có thư viện; sau đó khởi động lại kernel.
# %pip install "torch>=2.2,<3" "scikit-learn>=1.4,<2" "numpy>=1.26,<3" "pandas>=2.1,<3" "matplotlib>=3.8,<4" "pyarrow>=15"
import os, re, html, time, copy, random, json, tarfile, hashlib, platform
from pathlib import Path
from urllib.request import urlretrieve
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, ConfusionMatrixDisplay)
from IPython.display import display, Markdown

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(min(4, os.cpu_count() or 1))
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = Path('results'); OUT.mkdir(exist_ok=True)
print({'Python': platform.python_version(), 'PyTorch': torch.__version__,
       'sklearn': sklearn.__version__, 'device': str(DEVICE), 'seed': SEED})

DATA = Path('data'); DATA.mkdir(exist_ok=True)
frames, digest = {}, {}
for split in ['train','test']:
    path = DATA / f'{split}.parquet'
    if not path.exists():
        url = (f'https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/'
               f'plain_text/{split}-00000-of-00001.parquet')
        temporary = path.with_suffix('.download')
        urlretrieve(url,temporary)
        temporary.replace(path)
    digest[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    frames[split] = pd.read_parquet(path)[['text','label']]
    frames[split]['id'] = [f'{split}/{i}' for i in range(len(frames[split]))]
    assert len(frames[split]) == 25000
    assert frames[split]['label'].value_counts().to_dict() == {0:12500, 1:12500}
print('SHA256 dữ liệu:',digest)
raw_train, test = frames['train'], frames['test']
train, val = train_test_split(raw_train, test_size=0.2, random_state=SEED,
                             stratify=raw_train.label)
train, val = train.reset_index(drop=True), val.reset_index(drop=True)
assert set(train.id).isdisjoint(val.id)
assert set(raw_train.id).isdisjoint(test.id)
display(pd.DataFrame({s: f.label.value_counts().sort_index()
                     for s, f in [('train',train), ('validation',val), ('test',test)]}))

def clean_text(text):
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', text.lower()).strip()

def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[!?]", text)

for frame in (train, val, test): frame['clean'] = frame.text.map(clean_text)
lengths = train.clean.map(lambda s: len(tokenize(s)))
MAX_LEN, VOCAB_SIZE = 256, 20000
display(lengths.describe(percentiles=[.5,.9,.95]).to_frame('Số token / review (train)'))
print(f'Tỷ lệ review train bị cắt ở {MAX_LEN} token: {(lengths > MAX_LEN).mean():.2%}')
plt.figure(figsize=(8,3))
plt.hist(lengths, bins=60, range=(0,1200), color='#2563eb')
plt.axvline(MAX_LEN, color='#ef4444', linestyle='--', label=f'MAX_LEN={MAX_LEN}')
plt.xlabel('Số token'); plt.ylabel('Số review'); plt.title('Độ dài review trên tập train (hiển thị 0–1200)')
plt.legend(); plt.tight_layout(); plt.show()
display(train[['clean','label']].sample(2, random_state=SEED))

start = time.perf_counter()
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=2,
                        sublinear_tf=True, dtype=np.float32)
Xtr = tfidf.fit_transform(train.clean)
Xv = tfidf.transform(val.clean)
tfidf_seconds = time.perf_counter() - start
ytr, yv, ytest = train.label.to_numpy(), val.label.to_numpy(), test.label.to_numpy()
models, fit_times, selection, histories = {}, {}, [], {}
for name, cls in [('Logistic Regression',LogisticRegression), ('Linear SVM',LinearSVC)]:
    best_score, best_model = -1, None
    start = time.perf_counter()
    for C in [0.5, 2.0]:
        model = cls(C=C, max_iter=2000, random_state=SEED)
        model.fit(Xtr,ytr)
        score = f1_score(yv,model.predict(Xv))
        selection.append({'model':name,'C':C,'validation_F1':score})
        if score > best_score: best_score, best_model = score, model
    models[name] = best_model
    fit_times[name] = time.perf_counter()-start
display(pd.DataFrame(selection))
print('Số đặc trưng TF–IDF:', Xtr.shape[1], '| thời gian fit/transform train+val:', round(tfidf_seconds,2),'s')

counter = Counter(t for s in train.clean for t in tokenize(s))
vocab = {'<PAD>':0, '<UNK>':1}
vocab.update({w:i+2 for i,(w,_) in enumerate(counter.most_common(VOCAB_SIZE-2))})
def encode(texts):
    arr = np.zeros((len(texts),MAX_LEN),dtype=np.int64)
    lens = np.ones(len(texts),dtype=np.int64)
    for i,s in enumerate(texts):
        ids = [vocab.get(t,1) for t in tokenize(s)[:MAX_LEN]] or [1]
        arr[i,:len(ids)], lens[i] = ids,len(ids)
    return arr,lens
seq_tr,len_tr = encode(train.clean)
seq_v,len_v = encode(val.clean)

class MLP(nn.Module):
    def __init__(self,n_features):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features,64),nn.ReLU(),nn.Dropout(.5),nn.Linear(64,1))
    def forward(self,x,lengths=None): return self.net(x).squeeze(1)

class SentimentLSTM(nn.Module):
    def __init__(self,n_words):
        super().__init__()
        self.embedding = nn.Embedding(n_words,64,padding_idx=0)
        self.lstm = nn.LSTM(64,64,batch_first=True)
        self.dropout = nn.Dropout(.3)
        self.fc = nn.Linear(64,1)
    def forward(self,x,lengths):
        states,_ = self.lstm(self.embedding(x))
        last = states[torch.arange(x.size(0),device=x.device),lengths-1]
        return self.fc(self.dropout(last)).squeeze(1)

def batches(X,y=None,lengths=None,shuffle=False):
    order = np.arange(X.shape[0])
    if shuffle: np.random.shuffle(order)
    for begin in range(0,len(order),128):
        ix = order[begin:begin+128]
        if lengths is None:
            xb = torch.from_numpy(X[ix].toarray()).to(DEVICE)
            lb = None
        else:
            xb = torch.from_numpy(X[ix]).to(DEVICE)
            lb = torch.from_numpy(lengths[ix]).to(DEVICE)
        yb = None if y is None else torch.as_tensor(y[ix],dtype=torch.float32,device=DEVICE)
        yield xb,lb,yb

@torch.no_grad()
def neural_scores(model,X,lengths=None):
    model.eval()
    return np.concatenate([model(x,l).cpu().numpy() for x,l,_ in batches(X,lengths=lengths)])

def train_neural(name,model,X,Xval,lengths=None,val_lengths=None,max_epochs=8):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),lr=.001)
    loss_fn = nn.BCEWithLogitsLoss()
    best_f1, wait, best_state, history = -1,0,None,[]
    start = time.perf_counter()
    for epoch in range(1,max_epochs+1):
        model.train(); total_loss,total = 0.,0
        for xb,lb,yb in batches(X,ytr,lengths,shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb,lb),yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            total_loss += loss.item()*len(xb); total += len(xb)
        scores = neural_scores(model,Xval,val_lengths)
        vf1 = f1_score(yv,scores>=0)
        history.append({'epoch':epoch,'train_loss':total_loss/total,'validation_F1':vf1})
        print(f'{name} | epoch {epoch:02d} | loss {total_loss/total:.4f} | val F1 {vf1:.4f}',flush=True)
        if vf1 > best_f1:
            best_f1,wait = vf1,0
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 2: break
    model.load_state_dict(best_state)
    fit_times[name] = time.perf_counter()-start
    models[name],histories[name] = model,pd.DataFrame(history)
    print('Khôi phục checkpoint có validation F1 tốt nhất:',round(best_f1,4))

train_neural('MLP',MLP(Xtr.shape[1]),Xtr,Xv,max_epochs=8)
train_neural('LSTM',SentimentLSTM(len(vocab)),seq_tr,seq_v,len_tr,len_v,max_epochs=10)

Xt = tfidf.transform(test.clean)
seq_test,len_test = encode(test.clean)
rows, predictions, score_store = [],{},{}
for name,model in models.items():
    start = time.perf_counter()
    if name in ('MLP','LSTM'):
        scores = neural_scores(model, Xt if name=='MLP' else seq_test,
                               None if name=='MLP' else len_test)
    else:
        scores = model.decision_function(Xt)
    elapsed = time.perf_counter()-start
    pred = (scores>=0).astype(int)
    predictions[name],score_store[name] = pred,scores
    rows.append({'Model':name,'Accuracy':accuracy_score(ytest,pred),
                 'Precision':precision_score(ytest,pred,zero_division=0),
                 'Recall':recall_score(ytest,pred,zero_division=0),
                 'F1':f1_score(ytest,pred),'ROC_AUC':roc_auc_score(ytest,scores),
                 'Train_seconds':fit_times[name],'Predict_seconds':elapsed})
results = pd.DataFrame(rows).set_index('Model')
display(results.round(4))
results.to_csv(OUT/'comparison.csv',encoding='utf-8-sig')
pd.DataFrame(selection).to_csv(OUT/'validation_selection.csv',index=False)
for name,hist in histories.items(): hist.to_csv(OUT/f'{name}_history.csv',index=False)
metadata = {'seed':SEED,'device':str(DEVICE),'torch':str(torch.__version__),
            'sklearn':sklearn.__version__,'dataset_sha256':digest,
            'train':len(train),'validation':len(val),'test':len(test),
            'max_len':MAX_LEN,'vocabulary_size':len(vocab),
            'tfidf_features':Xtr.shape[1],'tfidf_preprocessing_seconds':tfidf_seconds}
(OUT/'run_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')

fig,axes = plt.subplots(1,2,figsize=(12,4))
results[['Accuracy','F1','ROC_AUC']].plot.bar(ax=axes[0],ylim=(0,1),rot=18)
axes[0].set_title('Chất lượng trên 25.000 review test'); axes[0].set_xlabel('')
for name,hist in histories.items():
    axes[1].plot(hist.epoch,hist.validation_F1,marker='o',label=name)
axes[1].set(xlabel='Epoch',ylabel='Validation F1',title='Chọn checkpoint bằng validation')
axes[1].legend(); plt.tight_layout(); plt.savefig(OUT/'comparison.png',dpi=160); plt.show()
fig,axes = plt.subplots(2,2,figsize=(9,8))
for ax,(name,pred) in zip(axes.ravel(),predictions.items()):
    ConfusionMatrixDisplay.from_predictions(ytest,pred,display_labels=['Tiêu cực','Tích cực'],
                                            ax=ax,colorbar=False,cmap='Blues')
    ax.set_title(name)
plt.tight_layout(); plt.savefig(OUT/'confusion_matrices.png',dpi=160); plt.show()
for name,pred in predictions.items():
    print('\n'+name)
    print(classification_report(ytest,pred,target_names=['Tiêu cực','Tích cực'],digits=4))

rng = np.random.default_rng(SEED)
diff = (predictions['LSTM']==ytest).astype(float)-(predictions['Linear SVM']==ytest).astype(float)
boot = np.array([rng.choice(diff,size=len(diff),replace=True).mean() for _ in range(2000)])
lo,hi = np.quantile(boot,[.025,.975])
print(f'Accuracy LSTM − SVM: {diff.mean():+.4f}; bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]')
wrong = np.flatnonzero(predictions['LSTM'] != ytest)
sample = rng.choice(wrong,size=min(6,len(wrong)),replace=False)
errors = test.iloc[sample][['id','clean','label']].copy()
errors['LSTM_prediction'] = predictions['LSTM'][sample]
errors['review_excerpt'] = errors.clean.str[:700]
display(errors[['id','review_excerpt','label','LSTM_prediction']].reset_index(drop=True))
errors.to_csv(OUT/'LSTM_error_examples.csv',index=False,encoding='utf-8-sig')

best = results.F1.idxmax()
lstm_f1,svm_f1 = results.loc['LSTM','F1'],results.loc['Linear SVM','F1']
display(Markdown(f"""### Nhận xét từ kết quả thực nghiệm

- Mô hình có test F1 cao nhất trong lần chạy: **{best}**, F1 **{results.loc[best,'F1']:.4f}**, accuracy **{results.loc[best,'Accuracy']:.4f}**. Đây là mô tả bảng kết quả, không dùng test để chỉnh mô hình.
- LSTM đạt F1 **{lstm_f1:.4f}**; SVM đạt **{svm_f1:.4f}**; chênh lệch F1 LSTM − SVM: **{lstm_f1-svm_f1:+.4f}**.
- LSTM mất **{fit_times['LSTM']:.1f} giây** huấn luyện trên **{DEVICE}**. TF–IDF chỉ cần nhận diện các từ/cụm từ mang cảm xúc; LSTM phải học cả embedding và cách tổng hợp chuỗi từ đầu.
- Có **{(lengths>MAX_LEN).mean():.1%}** review train dài hơn {MAX_LEN} token. Cắt đuôi có thể làm LSTM mất câu kết luận cảm xúc; TF–IDF sử dụng toàn văn nên có lợi thế thông tin đầu vào.
- Sai số có thể liên quan đến phủ định, mỉa mai, cảm xúc hỗn hợp hoặc nội dung bị cắt. Các khả năng này cần đối chiếu review đầy đủ, không thể khẳng định chỉ từ đoạn trích.

### Kết luận và hạn chế

Bài đã thực hiện LSTM và so sánh với mạng neural cơ bản MLP cùng hai phương pháp học máy Logistic Regression, SVM trên cùng phân chia IMDB. Chỉ có một seed và một cấu hình kiến trúc cho mỗi mạng; chưa thể khái quát rằng một họ mô hình luôn tốt hơn họ còn lại. Chưa đánh giá khả năng tổng quát sang miền khác hoặc tiếng Việt.

Hướng cải thiện: thử độ dài 512 token, BiLSTM hoặc embedding tiền huấn luyện; mở rộng tìm kiếm siêu tham số theo validation; chạy nhiều seed và báo cáo trung bình ± độ lệch chuẩn. Mỗi mở rộng phải giữ test độc lập, tránh điều chỉnh dựa trên kết quả test đã xem.
"""))

def predict_lstm(reviews):
    cleaned = [clean_text(s) for s in reviews]
    seq,lens = encode(cleaned)
    logits = neural_scores(models['LSTM'],seq,lens)
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    return pd.DataFrame({'review':reviews,'P(positive)':probs,
                         'prediction':np.where(probs>=.5,'Tích cực','Tiêu cực')})
display(predict_lstm([
    'A wonderful movie with excellent acting and a moving story.',
    'This movie was boring, badly written and a waste of time.',
    'The actors were good, but the story was disappointing.'
]))
print('Các bảng, biểu đồ và thông tin lần chạy đã lưu tại:',OUT.as_posix())
