# =========================================================
# evaluate.py
# =========================================================

import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
)


def evaluate_oof(
    y_true,
    y_prob,
    save_path,
):

    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)

    df = pd.DataFrame({

        "AUC": [auc],
        "ACC": [acc],

    })

    print(df)

    df.to_csv(save_path, index=False)
