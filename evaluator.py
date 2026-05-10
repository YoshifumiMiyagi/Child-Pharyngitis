import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score
)


class ThresholdEvaluator:
    def __init__(self, y_true, y_score, binary_thr=0.5):
        self.y_true = np.asarray(y_true)
        self.y_score = np.asarray(y_score)
        self.y_true_bin = (self.y_true >= binary_thr).astype(int)

        assert len(self.y_true_bin) == len(self.y_score)

    def overall_auc(self):
        return {
            "roc_auc": roc_auc_score(self.y_true_bin, self.y_score),
            "pr_auc": average_precision_score(self.y_true_bin, self.y_score),
        }

    def threshold_table(self, thresholds=None):
        if thresholds is None:
            thresholds = np.arange(0.05, 0.96, 0.05)

        rows = []

        for thr in thresholds:
            y_pred_bin = (self.y_score >= thr).astype(int)

            tn, fp, fn, tp = confusion_matrix(
                self.y_true_bin,
                y_pred_bin
            ).ravel()

            sen = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            spc = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

            rows.append({
                "threshold": round(thr, 2),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "SEN": sen,
                "SPC": spc,
                "PPV": ppv,
                "NPV": npv,
                "ACC": accuracy_score(self.y_true_bin, y_pred_bin),
                "BA": balanced_accuracy_score(self.y_true_bin, y_pred_bin),
                "F1": f1_score(self.y_true_bin, y_pred_bin, zero_division=0),
                "Youden_Index": sen + spc - 1
            })

        return pd.DataFrame(rows)
