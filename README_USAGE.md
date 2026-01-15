# RLF Prediction System - 使用說明

本系統包含兩個主要模組：
1. **Online Streaming KNN** - 串流資料即時預測 RLF 是否會在 0.9 秒內發生
2. **Embedding Model + Optuna** - 訓練嵌入模型並使用 Optuna 進行超參數搜尋

## 安裝依賴

```bash
pip install -r requirements.txt
```

## 環境設定

複製 `env.example` 到 `.env` 並填入您的設定值：

```bash
cp env.example .env
```

主要設定項目：

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `DATA_CSV_PATHS` | CSV 資料檔路徑（逗號分隔） | - |
| `OUTPUT_DIR` | 輸出目錄 | `results/optuna` |
| `N_TRIALS` | Optuna 試驗次數 | 50 |
| `MIN_FAR` | 最低誤報率約束 | 0.02 (2%) |
| `DELAY_SECONDS` | 延遲標記視窗（秒） | 0.9 |
| `KNN_MAX_SAMPLES` | KNN 最大儲存樣本數 | 500 |

---

## 1. Online Streaming KNN

### 時間單位

- **15625 frc = 1 秒**
- 延遲視窗預設 0.9 秒 = 14062.5 frc

### 基本用法

```python
from streaming_knn import OnlineStreamingKNN, KNNConfig, DistanceMetric, VotingScheme

# 建立設定
config = KNNConfig(
    max_samples=500,           # 最大儲存樣本數
    delay_seconds=0.9,         # 延遲視窗（秒）
    k_neighbors=5,             # K 值
    embedding_dim=16,          # 嵌入維度
    distance_metric=DistanceMetric.EUCLIDEAN,
    voting_scheme=VotingScheme.DISTANCE,
    rlf_class_weight=3.0,      # RLF 類別權重（處理不平衡）
    normal_class_weight=1.0,
    min_rlf_ratio=0.1,         # 維持最低 RLF 樣本比例
    max_rlf_ratio=0.5
)

# 建立 KNN
knn = OnlineStreamingKNN(config)
```

### 即時預測與更新

```python
import numpy as np

# 接收新樣本
embedding = model.get_embeddings(input_data)  # shape: (embedding_dim,)
current_timestamp = 1234567.0  # FRC 時間戳

# 1. 預測
prediction, confidence, details = knn.predict(embedding, current_timestamp)
# prediction: 0 = 無 RLF, 1 = 有 RLF
# confidence: 信心分數 [0, 1]
# details: 包含 rlf_probability, neighbor_labels 等資訊

# 2. 推入待處理佇列
knn.push(embedding, current_timestamp)

# 3. 在 0.9 秒後更新標籤
# 當知道真實標籤時呼叫
knn.update_label(original_timestamp, true_label)
```

### 批次處理（使用模擬器）

```python
from streaming_knn import StreamingKNNSimulator

# 建立模擬器
simulator = StreamingKNNSimulator(knn)

# 批次執行
embeddings = np.random.randn(100, 16).astype(np.float32)
labels = np.array([0, 1, 0, 0, 1, ...])  # 真實標籤

results = simulator.run_batch(
    embeddings, 
    labels, 
    time_step_seconds=0.3
)

print(f"Accuracy: {results['accuracy']}")
print(f"RLF Recall: {results['rlf_recall']}")
print(f"False Alarm Rate: {results['false_alarm_rate']}")
```

### 類別不平衡處理

KNN 使用多種策略處理 RLF 與 Normal 樣本的嚴重不平衡：

1. **類別權重** - `rlf_class_weight` 讓 RLF 樣本投票權重更高
2. **樣本池平衡** - 維持 `min_rlf_ratio` ~ `max_rlf_ratio` 的 RLF 比例
3. **時間衰減** - 較舊樣本權重較低
4. **距離反比投票** - 較近鄰居有更高投票權

---

## 2. Embedding Model + Optuna 超參數搜尋

### 模型架構

- **輸入**: 扁平化時間序列 (N × T × C → N × 350)，其中 T=35, C=10
- **嵌入層**: MLP 產生低維嵌入 (dim < 16)
- **分類頭**: 4 類別輸出

### 多重損失函數

1. **L_cls** - 加權交叉熵（處理類別不平衡）
2. **L_contrast** - 監督式對比損失（相同類別的嵌入更接近）
3. **L_time** - 時間感知損失（利用 `frc_diff` 資訊）

### 執行 Optuna 搜尋

```bash
# 使用合成資料進行測試
python embedding_optuna.py --use_synthetic --n_trials 10 --output_dir results/test

# 使用真實資料
python embedding_optuna.py \
    --csv_paths /path/to/data1.csv /path/to/data2.csv \
    --n_trials 50 \
    --min_far 0.02 \
    --output_dir results/optuna
```

### Python API

```python
from embedding_optuna import OptunaConfig, run_optuna_search

config = OptunaConfig(
    n_trials=50,
    min_far=0.02,          # FAR 需 >= 2%
    output_dir="results/optuna",
    batch_size=64,
    max_epochs=100,
    patience=10
)

study, summary = run_optuna_search(
    csv_paths=["/path/to/data1.csv", "/path/to/data2.csv"],
    config=config,
    use_synthetic=False
)

print(f"Best Trial: {summary['best_trial']}")
print(f"Best Recall: {summary['best_value']}")
print(f"Best Params: {summary['best_params']}")
```

### 超參數搜尋空間

| 參數 | 範圍 | 說明 |
|------|------|------|
| `embedding_dim` | 4-16 | 嵌入維度 |
| `n_layers` | 1-4 | 隱藏層數 |
| `hidden_*` | 32-256 | 每層神經元數 |
| `dropout_rate` | 0.1-0.5 | Dropout 比例 |
| `learning_rate` | 1e-5 ~ 1e-2 | 學習率 |
| `contrast_temp` | 0.05-0.5 | 對比損失溫度 |
| `time_loss_weight` | 0-0.5 | 時間損失權重 |
| `rlf_weight` | 1-10 | RLF 類別權重 |

### 輸出結構

```
results/optuna/
├── data_info.json          # 資料統計
├── study_summary.json      # 搜尋結果摘要
├── optimization_history.html
├── param_importances.html
└── trial_0000/
    ├── model_weights/      # 模型權重
    ├── metrics.json        # 評估指標
    ├── embedding_viz.html  # 嵌入視覺化（3D）
    ├── embedding_viz_2d.html
    ├── cm_2class.html      # 2類混淆矩陣
    └── cm_4class.html      # 4類混淆矩陣
```

### 嵌入視覺化

視覺化包含：
- **顏色**: 依類別區分 (RLF-0, RLF-1, RLF-2, Normal)
- **透明度**: 依 `time_diff` 調整（越接近 RLF 事件越不透明）
- **Hover**: 顯示標籤和時間差資訊

---

## 3. 完整範例

```bash
# 執行完整 Demo（KNN + Optuna + 推論流程）
python run_demo.py --mode full

# 僅執行 KNN Demo
python run_demo.py --mode knn

# 僅執行 Optuna Demo
python run_demo.py --mode optuna --n_trials 10
```

---

## 4. 資料格式

### 輸入 CSV 欄位

| 欄位 | 說明 |
|------|------|
| `frc_64us_10` | 當前樣本時間戳（frc 單位） |
| `frc_diff` | 距最近 RLF 事件的時間差（可能為 NaN） |
| `rlf_reason` | 標籤 (0/1/2 = RLF 類型, 3 = 正常) |
| 其他特徵 | 35 個時間步 × 10 個特徵 |

### 時間轉換

```python
FRC_PER_SECOND = 15625

# frc 轉秒
time_seconds = frc_value / FRC_PER_SECOND

# 秒轉 frc
frc_value = time_seconds * FRC_PER_SECOND
```

---

## 5. 評估指標

- **2-class Recall**: RLF 事件的偵測率 = TP / (TP + FN)
- **False Alarm Rate (FAR)**: 誤報率 = FP / (FP + TN)
- **Precision**: 精確率 = TP / (TP + FP)
- **F1 Score**: 2 × Precision × Recall / (Precision + Recall)

Optuna 目標：最大化 2-class Recall，約束 FAR >= k%（預設 2%）

---

## 疑難排解

### Q: KNN 預測都是 Normal？
A: 檢查 `rlf_class_weight` 是否足夠高，或增加初始 RLF 樣本

### Q: Optuna 收斂太慢？
A: 減少 `max_epochs`，增加 `patience`，或使用更小的搜尋空間

### Q: 記憶體不足？
A: 減少 `KNN_MAX_SAMPLES` 或 `batch_size`
