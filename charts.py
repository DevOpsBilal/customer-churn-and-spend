import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, confusion_matrix

SURF = "#fcfcfb"; INK = "#1c1c1c"; MUTED = "#6b6b6b"
BLUE = "#1F6FEB"; RED  = "#D1434F"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.edgecolor": "#d6d6d2", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": "#e8e8e4", "grid.linewidth": 0.8, "axes.axisbelow": True,
})

df = pd.read_csv('customer_engagement_practice.csv')
F = ['website_visits','minutes_on_site','emails_clicked','previous_spend','support_tickets']

# ---------- regression ----------
X, y = df[F], df['monthly_spend']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=42)
reg = LinearRegression().fit(Xtr, ytr)
pred = reg.predict(Xte)
mae, r2 = mean_absolute_error(yte, pred), r2_score(yte, pred)

fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=170)
lo, hi = 20, 200
ax.plot([lo, hi], [lo, hi], color=RED, lw=2, zorder=2, label="perfect prediction")
ax.scatter(yte, pred, s=52, color=BLUE, alpha=.85, zorder=3,
           edgecolor=SURF, linewidth=1.4, label="test customers")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("Actual monthly spend (£)", fontsize=11)
ax.set_ylabel("Predicted monthly spend (£)", fontsize=11)
ax.set_title("Linear Regression — predicted vs actual spend", fontsize=13.5, weight="bold", pad=14, loc="left")
ax.text(0.015, 0.965, f"MAE  £{mae:.2f}        R²  {r2:.2f}", transform=ax.transAxes,
        fontsize=11, color=MUTED, va="top")
ax.legend(fontsize=10, loc="lower right", frameon=True, facecolor=SURF, edgecolor="#e0e0dc")
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig("fig_regression.png"); plt.close()

# ---------- classification ----------
X, y = df[F], df['renewed_subscription']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
yp = clf.predict(Xte)
acc = accuracy_score(yte, yp); cm = confusion_matrix(yte, yp)

fig, ax = plt.subplots(figsize=(7.0, 5.4), dpi=170)
vmax = cm.max()
for i in range(2):
    for j in range(2):
        v = cm[i, j]
        shade = 0.10 + 0.78 * (v / vmax)          # one hue, light -> dark
        ax.add_patch(plt.Rectangle((j + .02, 1 - i + .02), .96, .96,
                     facecolor=BLUE, alpha=shade, edgecolor=SURF, linewidth=3))
        ax.text(j + .5, 1 - i + .56, str(v), ha="center", va="center",
                fontsize=30, weight="bold", color="#ffffff" if shade > .5 else INK)
        ax.text(j + .5, 1 - i + .28, "correct" if i == j else "wrong", ha="center",
                fontsize=10.5, color="#ffffff" if shade > .5 else MUTED)
ax.set_xlim(0, 2); ax.set_ylim(0, 2)
ax.set_xticks([.5, 1.5]); ax.set_xticklabels(["predicted: did not renew", "predicted: renewed"], fontsize=10.5)
ax.set_yticks([1.5, .5]); ax.set_yticklabels(["actually\ndid not renew", "actually\nrenewed"], fontsize=10.5)
ax.set_title("Confusion matrix — 36 test customers",
             fontsize=13.5, weight="bold", pad=34, loc="left")
ax.text(0, 2.07, f"Logistic Regression · test accuracy {acc:.1%}", fontsize=10.5, color=MUTED)
ax.grid(False)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
plt.tight_layout(); plt.savefig("fig_confusion.png"); plt.close()

# ---------- cross-validation ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(LogisticRegression(max_iter=1000), Xtr, ytr, cv=cv, scoring='accuracy')

fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=170)
xs = np.arange(5)
ax.bar(xs, scores, width=.58, color=BLUE, zorder=3)
# rules stop before the label gutter so nothing overlaps
ax.plot([-.6, 4.5], [scores.mean()]*2, color=RED, lw=2, zorder=4)
ax.plot([-.6, 4.5], [acc]*2, color=MUTED, lw=1.6, ls=(0, (5, 3)), zorder=4)
ax.text(4.68, acc, f"single split\n{acc:.3f}", color=MUTED, fontsize=10.5, va="center")
ax.text(4.68, scores.mean(), f"5-fold mean\n{scores.mean():.3f}", color=RED,
        fontsize=11, weight="bold", va="center")
for i, s in enumerate(scores):
    ax.text(i, s + .018, f"{s:.3f}", ha="center", fontsize=10.5, color=INK)
ax.set_xticks(xs); ax.set_xticklabels([f"fold {i+1}" for i in xs], fontsize=10.5)
ax.set_ylim(0, 1.0); ax.set_xlim(-.6, 6.4)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("One split flatters the model — five folds tell the truth",
             fontsize=13.5, weight="bold", pad=14, loc="left")
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig("fig_crossval.png"); plt.close()

print(f"MAE {mae:.2f}  R2 {r2:.2f}  acc {acc:.3f}  cm {cm.ravel()}  cv {scores.round(3)} mean {scores.mean():.3f}")
