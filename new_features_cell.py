# =====================================================================
# Phase 7: repeat clients + model-imputed EXT_SOURCE
# Paste as new cells AFTER the add_rates / add_payment_signals cell.
# Uses: train, test, preprocess, build_xy_fe, add_ratios, add_rates,
#       add_payment_signals, PARAMS, SEED (all already in main.ipynb)
# =====================================================================
from sklearn.model_selection import KFold

EXT = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]


def client_key(df):
    """Same person across applications.
    All DAYS_* columns are counted back from the application date, so their
    DIFFERENCES stay constant for the same person in different applications."""
    k1 = df["DAYS_BIRTH"] - df["DAYS_ID_PUBLISH"]
    k2 = (df["DAYS_BIRTH"] - df["DAYS_REGISTRATION"]).round()
    return df["CODE_GENDER"].astype(str) + "_" + k1.astype(str) + "_" + k2.astype(str)


def add_client(df):
    """Repeat-client features (no TARGET involved, safe on train + test)."""
    key = client_key(df)
    df["CLIENT_N_APPS"] = key.map(key.value_counts())
    df["_k"], df["_t"] = key, -df["DAYS_BIRTH"]          # smaller age = earlier application
    df["CLIENT_APP_RANK"] = df.groupby("_k")["_t"].rank(method="first")
    s = df.sort_values(["_k", "_t"])
    g = s.groupby("_k")
    df["DAYS_SINCE_PREV_APP"] = (s["_t"] - g["_t"].shift(1)).reindex(df.index)
    df["DAYS_TO_NEXT_APP"] = (g["_t"].shift(-1) - s["_t"]).reindex(df.index)
    df["PREV_APP_CREDIT_RATIO"] = (s["AMT_CREDIT"] / g["AMT_CREDIT"].shift(1)).reindex(df.index)
    df.drop(columns=["_k", "_t"], inplace=True)


def add_ext_pred(df):
    """Predict each EXT_SOURCE from the other columns (no TARGET involved).
    Rows that have the value get an out-of-fold prediction, rows without it get
    a prediction from a model trained on all rows that have it."""
    base = [c for c in df.columns if c not in ("SK_ID_CURR", "TARGET") and c not in EXT]
    Z = df[base].copy()
    for c in Z.select_dtypes(exclude="number").columns:
        Z[c] = Z[c].astype("category")
    reg = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, colsample_bytree=0.5, verbose=-1)
    for col in EXT:
        Zc = pd.concat([Z, df[[c for c in EXT if c != col]]], axis=1)
        has = df[col].notna().values
        idx, pred = np.where(has)[0], np.zeros(len(df))
        for tr, va in KFold(3, shuffle=True, random_state=0).split(idx):
            m = lgb.LGBMRegressor(**reg).fit(Zc.iloc[idx[tr]], df[col].values[idx[tr]])
            pred[idx[va]] = m.predict(Zc.iloc[idx[va]])
        m = lgb.LGBMRegressor(**reg).fit(Zc.iloc[idx], df[col].values[idx])
        pred[~has] = m.predict(Zc.iloc[np.where(~has)[0]])
        df[col + "_PRED"] = pred
        df[col + "_FILLED"] = df[col].fillna(pd.Series(pred, index=df.index))
    df["EXT_FILLED_MEAN"] = df[[c + "_FILLED" for c in EXT]].mean(axis=1)


# Client key for every row of train and test (same order as build_xy_fe)
_full_keys = client_key(pd.concat([train, test], ignore_index=True)).values
KEY_TR, KEY_TE = _full_keys[: len(train)], _full_keys[len(train):]


def other_apps_target(keys_fit, y_fit, keys_apply, exclude_self=False):
    """Mean TARGET of the client's OTHER applications, using only y_fit labels.
    exclude_self=True when keys_apply are the same rows as keys_fit (leave-one-out)."""
    kf, yv = pd.Series(keys_fit), pd.Series(np.asarray(y_fit, dtype=float))
    s, c = yv.groupby(kf.values).sum(), yv.groupby(kf.values).count()
    ka = pd.Series(keys_apply)
    tot, cnt = ka.map(s).fillna(0).values, ka.map(c).fillna(0).values
    if exclude_self:
        tot, cnt = tot - yv.values, cnt - 1
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def run_cv_v2(X, y, X_test, params=PARAMS, n_splits=5, verbose=True):
    """Same as run_cv, plus CLIENT_OTHER_TARGET computed INSIDE each fold
    (validation rows only see training-fold labels -> no leakage)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof, pred_test, fold_auc = np.zeros(len(X)), np.zeros(len(X_test)), []
    Xte = X_test.copy()
    Xte["CLIENT_OTHER_TARGET"] = other_apps_target(KEY_TR, y, KEY_TE)      # test: all train labels
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        Xtr, Xva = X.iloc[tr].copy(), X.iloc[va].copy()
        Xtr["CLIENT_OTHER_TARGET"] = other_apps_target(KEY_TR[tr], y.iloc[tr], KEY_TR[tr], exclude_self=True)
        Xva["CLIENT_OTHER_TARGET"] = other_apps_target(KEY_TR[tr], y.iloc[tr], KEY_TR[va])
        model = lgb.LGBMClassifier(**params)
        model.fit(Xtr, y.iloc[tr], eval_set=[(Xva, y.iloc[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = model.predict_proba(Xva)[:, 1]
        pred_test += model.predict_proba(Xte)[:, 1] / n_splits
        fold_auc.append(roc_auc_score(y.iloc[va], oof[va]))
        if verbose:
            print(f"Fold {fold + 1}: AUC {fold_auc[-1]:.4f}  ({model.best_iteration_} trees)")
    auc = roc_auc_score(y, oof)
    if verbose:
        print(f"OOF AUC: {auc:.4f}   (fold std: {np.std(fold_auc):.4f})")
    return dict(auc=auc, fold_std=np.std(fold_auc), oof=oof, pred_test=pred_test)


# ---------------------------------------------------------------- run it
X11, y11, X11_test = build_xy_fe([add_ratios, add_rates, add_payment_signals, add_client, add_ext_pred])
r11 = run_cv_v2(X11, y11, X11_test, params=p5)
