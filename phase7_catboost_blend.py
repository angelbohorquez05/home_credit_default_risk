# =====================================================================
# Phase 7b: CatBoost + LightGBM (3 seeds) blend on the new features
# Paste AFTER the cells from new_features_cell.py (needs X11, y11, X11_test,
# KEY_TR, KEY_TE, other_apps_target, run_cv_v2, p5, SEED, sample_sub)
# Runtime on 2 CPUs: ~17 min CatBoost + ~7 min LightGBM
# =====================================================================
from catboost import CatBoostClassifier


def run_cv_catboost_v2(X, y, X_test, n_splits=5):
    """CatBoost with CLIENT_OTHER_TARGET computed inside each fold (no leakage)."""
    cat_cols = X.select_dtypes(["category", "object"]).columns.tolist()
    X, X_test = X.copy(), X_test.copy()
    for c in cat_cols:                                   # CatBoost wants text categories without NaN
        X[c] = X[c].astype("object").fillna("MISSING").astype(str)
        X_test[c] = X_test[c].astype("object").fillna("MISSING").astype(str)
    X_test["CLIENT_OTHER_TARGET"] = other_apps_target(KEY_TR, y, KEY_TE)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof, pred_test, fold_auc = np.zeros(len(X)), np.zeros(len(X_test)), []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        Xtr, Xva = X.iloc[tr].copy(), X.iloc[va].copy()
        Xtr["CLIENT_OTHER_TARGET"] = other_apps_target(KEY_TR[tr], y.iloc[tr], KEY_TR[tr], exclude_self=True)
        Xva["CLIENT_OTHER_TARGET"] = other_apps_target(KEY_TR[tr], y.iloc[tr], KEY_TR[va])
        model = CatBoostClassifier(iterations=5000, learning_rate=0.05, depth=6, eval_metric="AUC",
                                   random_seed=SEED, od_type="Iter", od_wait=200, verbose=0,
                                   cat_features=cat_cols)
        model.fit(Xtr, y.iloc[tr], eval_set=(Xva, y.iloc[va]), use_best_model=True)
        oof[va] = model.predict_proba(Xva)[:, 1]
        pred_test += model.predict_proba(X_test[Xtr.columns])[:, 1] / n_splits
        fold_auc.append(roc_auc_score(y.iloc[va], oof[va]))
        print(f"Fold {fold + 1}: AUC {fold_auc[-1]:.4f}  ({model.get_best_iteration()} trees)")
    auc = roc_auc_score(y, oof)
    print(f"OOF AUC: {auc:.4f}   (fold std: {np.std(fold_auc):.4f})")
    return dict(auc=auc, fold_std=np.std(fold_auc), oof=oof, pred_test=pred_test)


r11_cb = run_cv_catboost_v2(X11, y11, X11_test)

# LightGBM: same folds, 3 model seeds, averaged
lgb_runs = [run_cv_v2(X11, y11, X11_test, params={**p5, "random_state": s}, verbose=False) for s in (42, 7, 2026)]
lgb_oof = np.mean([r["oof"] for r in lgb_runs], axis=0)
lgb_test = np.mean([r["pred_test"] for r in lgb_runs], axis=0)
print(f"LightGBM 3 seeds: {roc_auc_score(y11, lgb_oof):.4f} | CatBoost: {r11_cb['auc']:.4f}")

W = 0.5
blend_oof = W * lgb_oof + (1 - W) * r11_cb["oof"]
blend_test = W * lgb_test + (1 - W) * r11_cb["pred_test"]
print(f"Blend 50/50 CV AUC: {roc_auc_score(y11, blend_oof):.4f}")

sub = sample_sub.copy()
assert (sub["SK_ID_CURR"].values == test["SK_ID_CURR"].values).all()
sub["TARGET"] = blend_test
assert sub["TARGET"].between(0, 1).all() and sub["TARGET"].notna().all() and len(sub) == len(sample_sub)
sub.to_csv("output/submission_v7_blend_clients_extpred.csv", index=False)
