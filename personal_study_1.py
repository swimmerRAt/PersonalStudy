# 2.1 imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import kagglehub

# 2.2 재현성
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 2.3 데이터 로드 (여기를 당신의 경로/로더로 교체)
path = kagglehub.dataset_download("saurabhshahane/electricity-load-forecasting")
print("Path to dataset files:", path)

# [TODO] 여기를 실제 컬럼명으로 채우세요
feature_cols = ["Tmean", "RH", "AQI", "BaiduIndex", "Apmean"]  # 예시
target_cols  = ["ili_pct", "ili_pos_product"]                   # 다중 타깃 (단일이면 하나만)

# 2.4 하이퍼파라미터
SEQ_LEN = 12        # 12주 길이 입력
HORIZON = 1         # 1주 앞 예측
LSTM_UNITS = 64
BATCH_SIZE = 64
EPOCHS = 100
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 2.5 시계열 분할 인덱스 계산
def time_series_split_indexes(n, val_ratio=0.15, test_ratio=0.15):
    test_size = int(n * test_ratio)
    val_size  = int(n * val_ratio)
    train_end = n - (val_size + test_size)
    val_end   = n - test_size
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)

# 2.6 윈도우 생성 함수
def make_windows(X_2d, y_2d, seq_len=12, horizon=1):
    """
    X_2d: [T, F], y_2d: [T, O]  (시간축 T, 피처 F, 출력 O)
    반환: X:[N, seq_len, F], y:[N, O]
    """
    T = X_2d.shape[0]
    N = T - seq_len - horizon + 1
    if N <= 0:
        return (np.zeros((0, seq_len, X_2d.shape[1]), dtype=np.float32),
                np.zeros((0, y_2d.shape[1]), dtype=np.float32))
    X = np.zeros((N, seq_len, X_2d.shape[1]), dtype=np.float32)
    y = np.zeros((N, y_2d.shape[1]), dtype=np.float32)
    for i in range(N):
        X[i] = X_2d[i:i+seq_len]
        y[i] = y_2d[i+seq_len+horizon-1]
    return X, y

# 2.7 스케일링(Train만 fit)
def scale_by_train(train_arr, val_arr, test_arr):
    scaler = MinMaxScaler()
    scaler.fit(train_arr)
    return scaler, scaler.transform(train_arr), scaler.transform(val_arr), scaler.transform(test_arr)

# 2.8 DataFrame → 넘파이
def df_to_arrays(df, feature_cols, target_cols):
    X_all = df[feature_cols].values.astype(np.float32)
    y_all = df[target_cols].values.astype(np.float32)
    return X_all, y_all

# 2.9 메인 준비 루틴
def prepare_tensors(df, feature_cols, target_cols, seq_len=12, horizon=1, val_ratio=0.15, test_ratio=0.15):
    X_all, y_all = df_to_arrays(df, feature_cols, target_cols)
    n = len(df)
    tr_idx, va_idx, te_idx = time_series_split_indexes(n, val_ratio, test_ratio)

    X_tr_raw, y_tr_raw = X_all[tr_idx], y_all[tr_idx]
    X_va_raw, y_va_raw = X_all[va_idx], y_all[va_idx]
    X_te_raw, y_te_raw = X_all[te_idx], y_all[te_idx]

    # 스케일러는 feature/target 각각 별도로
    f_scaler, X_tr, X_va, X_te = scale_by_train(X_tr_raw, X_va_raw, X_te_raw)
    t_scaler, y_tr, y_va, y_te = scale_by_train(y_tr_raw, y_va_raw, y_te_raw)

    # 윈도우 생성(주의: 각 split 내부에서 생성)
    X_tr_w, y_tr_w = make_windows(X_tr, y_tr, seq_len, horizon)
    X_va_w, y_va_w = make_windows(X_va, y_va, seq_len, horizon)
    X_te_w, y_te_w = make_windows(X_te, y_te, seq_len, horizon)

    return (X_tr_w, y_tr_w, X_va_w, y_va_w, X_te_w, y_te_w, f_scaler, t_scaler)

# 2.10 LSTM 모델
def build_lstm(input_shape, output_dim, units=64, dropout=0.2):
    inp = layers.Input(shape=input_shape)                 # (seq_len, F)
    x   = layers.LSTM(units, return_sequences=False)(inp) # 마지막 타임스텝 출력만 사용
    x   = layers.Dropout(dropout)(x)
    out = layers.Dense(output_dim, activation="linear")(x)
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="mse",
                  metrics=["mae", "mse"])
    return model

# 2.11 학습 + 평가 함수
def train_and_evaluate(df, feature_cols, target_cols):
    X_tr, y_tr, X_va, y_va, X_te, y_te, f_scaler, t_scaler = prepare_tensors(
        df, feature_cols, target_cols, seq_len=SEQ_LEN, horizon=HORIZON,
        val_ratio=VAL_RATIO, test_ratio=TEST_RATIO
    )

    model = build_lstm(input_shape=(SEQ_LEN, len(feature_cols)),
                       output_dim=len(target_cols),
                       units=LSTM_UNITS, dropout=0.2)

    es = callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)

    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[es],
        shuffle=False  # 시계열이므로 섞지 않음
    )

    # 테스트 예측 (스케일 역변환하여 실제 스케일에서 지표 계산)
    y_pred_scaled = model.predict(X_te)
    # 역변환하려면 2D가 필요: [N, out_dim]
    y_pred = t_scaler.inverse_transform(y_pred_scaled)
    y_true = t_scaler.inverse_transform(y_te)

    metrics = {
        "R2": r2_score(y_true, y_pred, multioutput="variance_weighted"),
        "ExplainedVariance": explained_variance_score(y_true, y_pred, multioutput="variance_weighted"),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
    }
    return model, metrics, hist

# ====== FERRET DATA PIPELINE (add below your code) ======
from sklearn.model_selection import GroupShuffleSplit
import os, glob

# (A) CSV 로드 + 그룹별 시계열 정리
def load_ferret_df(inputs_dir="inputs", filename_glob="*.csv"):
    """
    ferret 레포의 inputs/*.csv를 읽어 하나의 DataFrame으로 합침.
    필요한 컬럼이 실제 파일과 다를 수 있으니, 아래 후보 중 존재하는 것만 사용.
    """
    files = sorted(glob.glob(os.path.join(inputs_dir, filename_glob)))
    if not files:
        raise FileNotFoundError(f"No CSV files under {inputs_dir}/{filename_glob}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    # ---- 컬럼 후보 (실제 파일에서 존재하는 것만 사용) ----
    id_cols_candidates   = ["study_id", "virus", "animal_id"]
    time_col_candidates  = ["day_post_inoculation", "dpi", "day"]
    feat_candidates_pool = [
        "percent_weight_change", "body_temp", "clinical_score",
        "nasal_wash_titer_log10", "titer_log10", "fever_score"
    ]
    target_candidates    = ["percent_weight_change"]  # 회귀 타깃 예시(다음 날 체중변화 예측)

    # --- 실제 사용 컬럼 결정 ---
    id_cols = [c for c in id_cols_candidates if c in df.columns]
    if len(id_cols) < 2:
        raise ValueError(f"ID columns not found sufficiently in {df.columns.tolist()}")

    time_col = next((c for c in time_col_candidates if c in df.columns), None)
    if time_col is None:
        raise ValueError(f"Time column not found in {df.columns.tolist()}")

    feat_cols = [c for c in feat_candidates_pool if c in df.columns]
    if not feat_cols:
        raise ValueError("No feature columns found. Check your CSVs.")

    # 타깃은 일단 percent_weight_change 가정(존재 시)
    target_cols = [c for c in target_candidates if c in df.columns]
    if not target_cols:
        # 없으면 특징 중 하나를 선택해도 되지만, 여기선 명시적으로 요구
        raise ValueError("Target column percent_weight_change not found.")

    # 정렬
    df = df.sort_values(id_cols + [time_col]).reset_index(drop=True)

    # 결측/형 변환(간단 처리)
    use_cols = id_cols + [time_col] + feat_cols + target_cols
    df = df[use_cols].copy()

    # dpi가 정수라고 가정, 일부 결측 있으면 채움
    # 그룹별로 0..max_dpi까지 reindex → ffill → 0 채움
    filled = []
    for keys, g in df.groupby(id_cols, dropna=False):
        g = g.sort_values(time_col)
        min_d = int(g[time_col].min())
        max_d = int(g[time_col].max())
        full_index = pd.Index(range(min_d, max_d + 1), name=time_col)
        g2 = g.set_index(time_col).reindex(full_index)

        # 메타/ID 복구
        for i, c in enumerate(id_cols):
            g2[c] = keys[i]

        # 수치 컬럼은 ffill 후 0
        for c in feat_cols + target_cols:
            if c in g2.columns:
                g2[c] = g2[c].astype(float)

        g2[feat_cols + target_cols] = g2[feat_cols + target_cols].ffill().fillna(0.0)
        g2 = g2.reset_index()
        filled.append(g2)

    df_full = pd.concat(filled, ignore_index=True)

    # 최종 선택/정렬
    df_full = df_full[id_cols + [time_col] + feat_cols + target_cols]
    df_full = df_full.sort_values(id_cols + [time_col]).reset_index(drop=True)

    return df_full, id_cols, time_col, feat_cols, target_cols

# (B) 그룹 기반 split → 스케일링 → 각 split에서 윈도우링
def prepare_tensors_ferret(
    df_full, id_cols, time_col, feature_cols, target_cols,
    seq_len=12, horizon=1, group_by="virus", val_ratio=0.15, test_ratio=0.15
):
    """
    ferret 전용: 같은 group(virus 등)이 train/val/test에 섞이지 않도록 group split.
    스케일러는 train에서만 fit. 각 split 내부에서 윈도우링.
    """
    if group_by not in df_full.columns:
        raise ValueError(f"group_by='{group_by}' column not in DataFrame.")

    # --- 그룹 단위 목록 ---
    # group 단위(virus 추천), 그 안에 animal 여러 개
    groups = df_full[[group_by]].drop_duplicates().reset_index(drop=True)
    group_vals = groups[group_by].values

    gss = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
    idx_all = np.arange(len(group_vals))
    tr_idx_g, te_idx_g = next(gss.split(idx_all, groups=group_vals))

    # val split은 train 그룹 내에서 다시
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio/(1.0 - test_ratio), random_state=42)
    tr_idx_g2, va_idx_g2 = next(gss2.split(tr_idx_g, groups=group_vals[tr_idx_g]))

    # 실제 그룹 값
    tr_groups = set(group_vals[tr_idx_g][tr_idx_g2])
    va_groups = set(group_vals[tr_idx_g][va_idx_g2])
    te_groups = set(group_vals[te_idx_g])

    def _subset_by_groups(df, group_values):
        return df[df[group_by].isin(group_values)].copy()

    df_tr = _subset_by_groups(df_full, tr_groups)
    df_va = _subset_by_groups(df_full, va_groups)
    df_te = _subset_by_groups(df_full, te_groups)

    # --- 스케일링: train만 fit ---
    # 2D array로 fit (시간축 전체 펼쳐서)
    X_tr_2d = df_tr[feature_cols].values.astype(np.float32)
    y_tr_2d = df_tr[target_cols].values.astype(np.float32)

    f_scaler = MinMaxScaler().fit(X_tr_2d)
    t_scaler = MinMaxScaler().fit(y_tr_2d)

    def _transform(df):
        X = f_scaler.transform(df[feature_cols].values.astype(np.float32))
        y = t_scaler.transform(df[target_cols].values.astype(np.float32))
        df2 = df.copy()
        df2[feature_cols] = X
        df2[target_cols]  = y
        return df2

    df_tr_s = _transform(df_tr)
    df_va_s = _transform(df_va)
    df_te_s = _transform(df_te)

    # --- 각 split에서 animal 단위로 윈도우링 ---
    def _windows_from_split(df_s):
        X_list, y_list = [], []
        # animal_id가 없다면 (study_id, virus) 조합을 사용
        g_cols = id_cols if "animal_id" in id_cols else list(dict.fromkeys(id_cols + [group_by]))
        for _, g in df_s.groupby(g_cols, dropna=False):
            g = g.sort_values(time_col)
            X_2d = g[feature_cols].values.astype(np.float32)   # [T, F]
            y_2d = g[target_cols].values.astype(np.float32)    # [T, O]
            if X_2d.shape[0] < (seq_len + horizon):
                continue
            X_w, y_w = make_windows(X_2d, y_2d, seq_len=seq_len, horizon=horizon)
            X_list.append(X_w)
            y_list.append(y_w)
        if not X_list:
            return (np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32),
                    np.zeros((0, len(target_cols)), dtype=np.float32))
        return np.vstack(X_list), np.vstack(y_list)

    X_tr_w, y_tr_w = _windows_from_split(df_tr_s)
    X_va_w, y_va_w = _windows_from_split(df_va_s)
    X_te_w, y_te_w = _windows_from_split(df_te_s)

    return (X_tr_w, y_tr_w, X_va_w, y_va_w, X_te_w, y_te_w, f_scaler, t_scaler, df_tr, df_va, df_te)

# (C) 학습/평가 (ferret 전용)
def train_and_evaluate_ferret(
    inputs_dir="inputs", filename_glob="*.csv",
    seq_len=12, horizon=1, group_by="virus",
    lstm_units=64, dropout=0.2, batch_size=64, epochs=100, val_ratio=0.15, test_ratio=0.15
):
    # 1) 로드
    df_full, id_cols, time_col, feature_cols, target_cols = load_ferret_df(inputs_dir, filename_glob)

    # 2) 텐서 준비
    (X_tr, y_tr, X_va, y_va, X_te, y_te,
     f_scaler, t_scaler, df_tr_raw, df_va_raw, df_te_raw) = prepare_tensors_ferret(
        df_full, id_cols, time_col, feature_cols, target_cols,
        seq_len=seq_len, horizon=horizon, group_by=group_by,
        val_ratio=val_ratio, test_ratio=test_ratio
    )

    # 3) 모델
    model = build_lstm(input_shape=(seq_len, len(feature_cols)),
                       output_dim=len(target_cols),
                       units=lstm_units, dropout=dropout)

    es = callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)

    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        shuffle=False
    )

    # 4) 평가(역스케일)
    y_pred_scaled = model.predict(X_te)
    y_pred = t_scaler.inverse_transform(y_pred_scaled)
    y_true = t_scaler.inverse_transform(y_te)

    metrics = {
        "R2": r2_score(y_true, y_pred, multioutput="variance_weighted"),
        "ExplainedVariance": explained_variance_score(y_true, y_pred, multioutput="variance_weighted"),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "n_train_windows": int(X_tr.shape[0]),
        "n_val_windows": int(X_va.shape[0]),
        "n_test_windows": int(X_te.shape[0]),
        "n_train_groups": int(df_tr_raw[group_by].nunique()),
        "n_val_groups": int(df_va_raw[group_by].nunique()),
        "n_test_groups": int(df_te_raw[group_by].nunique()),
    }
    return model, metrics, hist, (feature_cols, target_cols)

# ferret 데이터로 학습/평가
model, metrics, hist, (feature_cols, target_cols) = train_and_evaluate_ferret(
    inputs_dir="machine-learning-influenza-ferret-model/inputs",  # 레포 경로에 맞춰 수정
    filename_glob="*.csv",
    seq_len=5,           # ferret 일(day) 단위라면 5일 윈도우부터 시작 권장
    horizon=1,           # 1일 뒤 예측
    group_by="virus",    # 누수 방지용 그룹(권장). 필요시 "study_id"로 바꿔도 됨
    lstm_units=64,
    dropout=0.2,
    batch_size=64,
    epochs=100,
    val_ratio=0.15,
    test_ratio=0.15
)
print(metrics)
