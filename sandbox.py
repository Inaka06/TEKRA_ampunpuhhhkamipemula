"""
Model 3 — MH Untreated Prediction (FIXED: OneHotEncoder untuk course_group)
============================================================================
Target : untreated_mh = punya MH issue AND belum cari bantuan
Fitur  : demografi saja (gender, age, year, cgpa, married, course_group)
Fix    : course_group pakai OneHotEncoder (nominal, bukan ordinal)
"""

import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (StratifiedKFold, cross_validate,
    cross_val_predict, GridSearchCV)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, roc_curve, f1_score)

# ──────────────────────────────────────────────
# 1. LOAD & FEATURE ENGINEERING
# ──────────────────────────────────────────────
df = pd.read_csv("Student_Mental_health.csv")

def engineer(df):
    d = df.copy()
    d.drop(columns=["Timestamp"], inplace=True)
    d.rename(columns={
        "Choose your gender":"gender","Age":"age","What is your course?":"course",
        "Your current year of Study":"year","What is your CGPA?":"cgpa",
        "Marital status":"married","Do you have Depression?":"depression",
        "Do you have Anxiety?":"anxiety","Do you have Panic attack?":"panic",
        "Did you seek any specialist for a treatment?":"treatment",
    }, inplace=True)
    for col in d.select_dtypes("object").columns:
        d[col] = d[col].str.strip().str.lower()
    d["year"] = d["year"].map({"year 1":1,"year 2":2,"year 3":3,"year 4":4}).fillna(1).astype(int)
    cgpa_map = {"0 - 1.99":0,"2.00 - 2.49":1,"2.50 - 2.99":2,"3.00 - 3.49":3,"3.50 - 4.00":4}
    d["cgpa"] = d["cgpa"].map(cgpa_map).fillna(2).astype(int)  # CGPA tetap ordinal (ada urutan!)

    def group_course(c):
        for k in ["engineering","bit","bcs","koe","it","cts","enm","econs","kenms",
                  "biotechnology","marine","radiography","mhsc","kop","malcom"]:
            if k in c: return "stem"
        for k in ["biomedical","nursing","medical"]:
            if k in c: return "health"
        for k in ["banking","business","accounting"]:
            if k in c: return "business"
        return "humanities"

    d["course_group"] = d["course"].apply(group_course)
    d.drop(columns=["course"], inplace=True)
    d["gender"]    = df["Choose your gender"].str.strip().str.lower().map({"male":1,"female":0}).values
    d["married"]   = (d["married"]=="yes").astype(int)
    d["depression"]= (d["depression"]=="yes").astype(int)
    d["anxiety"]   = (d["anxiety"]=="yes").astype(int)
    d["panic"]     = (d["panic"]=="yes").astype(int)
    d["treatment"] = (d["treatment"]=="yes").astype(int)
    d["age"]       = d["age"].fillna(d["age"].median())
    return d

df_c = engineer(df)

# ──────────────────────────────────────────────
# 2. TARGET: untreated_mh
# ──────────────────────────────────────────────
any_mh = ((df_c.depression + df_c.anxiety + df_c.panic) > 0).astype(int)
df_c["untreated_mh"] = ((any_mh == 1) & (df_c.treatment == 0)).astype(int)

print("Target distribution:")
print(df_c["untreated_mh"].value_counts().rename({0:"Aman (kelas 0)", 1:"MH Untreated (kelas 1)"}).to_string())

# ──────────────────────────────────────────────
# 3. FITUR & PREPROCESSOR
# ──────────────────────────────────────────────
FEATS    = ["gender", "age", "year", "cgpa", "married", "course_group"]
num_cols = ["gender", "age", "year", "cgpa", "married"]
cat_cols = ["course_group"]

X = df_c[FEATS]
y = df_c["untreated_mh"]

# course_group = nominal → OneHotEncoder (drop='first' hindari dummy trap)
# cgpa = ordinal → sudah di-encode sebagai int, masuk num pipeline
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
    ]), num_cols),
    ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols),
])

# ──────────────────────────────────────────────
# 4. CV + KANDIDAT MODEL
# ──────────────────────────────────────────────
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
    "GBM":                 GradientBoostingClassifier(n_estimators=100, learning_rate=0.03, max_depth=2, random_state=42),
    "SVM (RBF)":           SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
}

print("\n── Cross-Validation Results ──")
all_results = {}
for name, clf in candidates.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    s = cross_validate(pipe, X, y, cv=cv5,
        scoring=["f1","roc_auc","average_precision","precision","recall"],
        return_train_score=True)
    all_results[name] = {
        "F1":      s["test_f1"].mean(),
        "F1_std":  s["test_f1"].std(),
        "AUC":     s["test_roc_auc"].mean(),
        "AP":      s["test_average_precision"].mean(),
        "Prec":    s["test_precision"].mean(),
        "Rec":     s["test_recall"].mean(),
        "Train_F1":s["train_f1"].mean(),
    }
    r = all_results[name]
    print(f"  {name:22s} | F1={r['F1']:.3f}±{r['F1_std']:.3f}  AUC={r['AUC']:.3f}"
          f"  AP={r['AP']:.3f}  Prec={r['Prec']:.3f}  Rec={r['Rec']:.3f}"
          f"  [overfit={r['Train_F1']-r['F1']:+.3f}]")

# ──────────────────────────────────────────────
# 5. HYPERPARAMETER TUNING
# ──────────────────────────────────────────────
print("\n── Grid Search ──")

rf_pipe = Pipeline([("prep", preprocessor),
                    ("clf", RandomForestClassifier(class_weight="balanced", random_state=42))])
gs_rf = GridSearchCV(rf_pipe,
    {"clf__max_depth":[3,5,None],"clf__min_samples_leaf":[1,2],
     "clf__n_estimators":[100,200],"clf__max_features":["sqrt","log2"]},
    cv=cv5, scoring="roc_auc", n_jobs=-1)
gs_rf.fit(X, y)
print(f"  RF  best AUC={gs_rf.best_score_:.4f}  {gs_rf.best_params_}")

gbm_pipe = Pipeline([("prep", preprocessor),
                     ("clf", GradientBoostingClassifier(random_state=42))])
gs_gbm = GridSearchCV(gbm_pipe,
    {"clf__learning_rate":[0.03,0.05,0.1],"clf__max_depth":[2,3],
     "clf__n_estimators":[100,200],"clf__subsample":[0.8,1.0]},
    cv=cv5, scoring="roc_auc", n_jobs=-1)
gs_gbm.fit(X, y)
print(f"  GBM best AUC={gs_gbm.best_score_:.4f}  {gs_gbm.best_params_}")

# ──────────────────────────────────────────────
# 6. ENSEMBLE (Soft Voting)
# ──────────────────────────────────────────────
lr_pipe = Pipeline([("prep", preprocessor),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])

ensemble = VotingClassifier(
    estimators=[("lr", lr_pipe), ("rf", gs_rf.best_estimator_), ("gbm", gs_gbm.best_estimator_)],
    voting="soft", weights=[1, 2, 2]
)

ens_s = cross_validate(ensemble, X, y, cv=cv5,
    scoring=["f1","roc_auc","average_precision","precision","recall"])
print(f"\n  Ensemble: F1={ens_s['test_f1'].mean():.3f}±{ens_s['test_f1'].std():.3f}"
      f"  AUC={ens_s['test_roc_auc'].mean():.3f}"
      f"  AP={ens_s['test_average_precision'].mean():.3f}")

# ──────────────────────────────────────────────
# 7. FINAL PREDICTIONS + OPTIMAL THRESHOLD
# ──────────────────────────────────────────────
y_pred  = cross_val_predict(ensemble, X, y, cv=cv5, method="predict")
y_proba = cross_val_predict(ensemble, X, y, cv=cv5, method="predict_proba")[:, 1]

prec_arr, rec_arr, thr_arr = precision_recall_curve(y, y_proba)
f1_arr   = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)
best_idx = np.argmax(f1_arr[:-1])
best_thr = thr_arr[best_idx]
y_opt    = (y_proba >= best_thr).astype(int)

print(f"\n  Optimal threshold: {best_thr:.3f}")
print("\n── Final Report @ Optimal Threshold ──")
print(classification_report(y, y_opt, target_names=["Aman", "MH Untreated"]))
print(f"  ROC-AUC : {roc_auc_score(y, y_proba):.4f}")
print(f"  Avg Prec: {average_precision_score(y, y_proba):.4f}")

# ──────────────────────────────────────────────
# 8. FEATURE IMPORTANCE
# ──────────────────────────────────────────────
ensemble.fit(X, y)

ohe_feat_names = (ensemble.estimators_[0]
    .named_steps["prep"]
    .named_transformers_["cat"]
    .get_feature_names_out(["course_group"]).tolist())
all_feat_names = num_cols + ohe_feat_names  # ['gender','age','year','cgpa','married','course_group_health',...]

rf_arm  = ensemble.estimators_[1].named_steps["clf"]
gbm_arm = ensemble.estimators_[2].named_steps["clf"]
imp_avg = pd.Series(
    (rf_arm.feature_importances_ + gbm_arm.feature_importances_) / 2,
    index=all_feat_names
).sort_values(ascending=False)

print("\n── Feature Importance (avg RF + GBM) ──")
print(imp_avg.round(4).to_string())

# Risk score per subgroup
df_c["risk_score"] = y_proba
print("\n── Risk Score by Course Group ──")
print(df_c.groupby("course_group")["risk_score"].agg(["mean","count"]).round(3))

# ──────────────────────────────────────────────
# 9. VISUALISASI
# ──────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
BLUE, ORG, RED, GRN = "#4C72B0", "#DD8452", "#C44E52", "#55A868"

fig = plt.figure(figsize=(18, 13))
fig.suptitle("Model 3 — MH Untreated Prediction (OneHotEncoder Fix)",
             fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

# Panel 1: Target distribution
ax = fig.add_subplot(gs[0, 0])
vals   = [y.value_counts()[0], y.value_counts()[1]]
bars   = ax.bar(["Aman\n(kelas 0)", "MH Untreated\n(kelas 1)"], vals,
                color=[GRN, RED], edgecolor="white", linewidth=1.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.5, str(v), ha="center", fontsize=12, fontweight="bold")
ax.set_title("Distribusi Target", fontweight="bold"); ax.set_ylim(0, 75)

# Panel 2: CV scores semua model
ax = fig.add_subplot(gs[0, 1])
res_df = pd.DataFrame(all_results).T[["F1","AUC","AP"]].copy()
res_df.loc["Ensemble"] = [ens_s["test_f1"].mean(), ens_s["test_roc_auc"].mean(),
                           ens_s["test_average_precision"].mean()]
res_df.plot(kind="bar", ax=ax, rot=25, colormap="Set2", edgecolor="white")
ax.set_title("CV Scores — Semua Kandidat", fontweight="bold")
ax.set_ylim(0.4, 0.9); ax.legend(fontsize=8); ax.tick_params(axis="x", labelsize=8)

# Panel 3: Feature importance (OHE — nama kolom jelas)
ax = fig.add_subplot(gs[0, 2])
imp_avg.sort_values().plot(kind="barh", ax=ax, color=BLUE, edgecolor="white")
ax.set_title("Feature Importance\n(avg RF+GBM, OHE names)", fontweight="bold")
ax.set_xlabel("Importance"); ax.invert_yaxis()

# Panel 4: Confusion matrix
ax = fig.add_subplot(gs[1, 0])
cm = confusion_matrix(y, y_opt)
ConfusionMatrixDisplay(cm, display_labels=["Aman","MH Untreated"]).plot(ax=ax, colorbar=False, cmap="RdYlGn_r")
ax.set_title(f"Confusion Matrix\n@ threshold={best_thr:.2f}", fontweight="bold")

# Panel 5: ROC Curve
ax = fig.add_subplot(gs[1, 1])
fpr, tpr, _ = roc_curve(y, y_proba)
auc_val = roc_auc_score(y, y_proba)
ax.plot(fpr, tpr, color=BLUE, lw=2.5, label=f"AUC = {auc_val:.3f}")
ax.fill_between(fpr, tpr, alpha=0.12, color=BLUE)
ax.plot([0,1],[0,1],"k--",lw=1)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC Curve", fontweight="bold"); ax.legend()

# Panel 6: Precision-Recall
ax = fig.add_subplot(gs[1, 2])
ap_val = average_precision_score(y, y_proba)
ax.plot(rec_arr, prec_arr, color=ORG, lw=2.5, label=f"AP = {ap_val:.3f}")
ax.fill_between(rec_arr, prec_arr, alpha=0.12, color=ORG)
ax.scatter([rec_arr[best_idx]], [prec_arr[best_idx]], s=100, color=RED, zorder=5,
           label=f"Optimal thr={best_thr:.2f}  F1={f1_arr[best_idx]:.3f}")
ax.axhline(y.mean(), color="gray", linestyle="--", lw=1, label=f"Baseline ({y.mean():.2f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve", fontweight="bold"); ax.legend(fontsize=8)

# Panel 7: Risk by Year
ax = fig.add_subplot(gs[2, 0])
yr_risk = df_c.groupby("year")["risk_score"].mean()
yr_risk.index = [f"Year {i}" for i in yr_risk.index]
yr_risk.plot(kind="bar", ax=ax, color=ORG, edgecolor="white", rot=0)
ax.axhline(y_proba.mean(), color="red", linestyle="--", lw=1.5, label="Overall avg")
ax.set_title("Risk Score per Tahun Studi", fontweight="bold")
ax.set_ylabel("Avg Risk Score"); ax.set_ylim(0, 1); ax.legend()

# Panel 8: Risk by Course Group
ax = fig.add_subplot(gs[2, 1])
cg_risk = df_c.groupby("course_group")["risk_score"].mean().sort_values()
cg_risk.plot(kind="barh", ax=ax, color=BLUE, edgecolor="white")
ax.axvline(y_proba.mean(), color="red", linestyle="--", lw=1.5)
ax.set_title("Risk Score per Course Group", fontweight="bold")
ax.set_xlabel("Avg Risk Score"); ax.set_xlim(0, 1)

# Panel 9: Risk score distribution per kelas aktual
ax = fig.add_subplot(gs[2, 2])
for lbl, color, name in [(0, GRN, "Aman"), (1, RED, "MH Untreated")]:
    ax.hist(y_proba[y==lbl], bins=12, alpha=0.6, color=color,
            label=f"{name} (n={int((y==lbl).sum())})", edgecolor="white")
ax.axvline(best_thr, color="black", linestyle="--", lw=1.5, label=f"Threshold={best_thr:.2f}")
ax.set_xlabel("Risk Score"); ax.set_ylabel("Frekuensi")
ax.set_title("Distribusi Risk Score per Kelas", fontweight="bold"); ax.legend(fontsize=8)

fig.savefig("model3_ohe_fixed.png", dpi=150, bbox_inches="tight")
print("\n✅ Saved → model3_ohe_fixed.png")