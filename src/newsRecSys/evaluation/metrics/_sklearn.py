try:
    from sklearn.metrics import (
        # _classification:
        accuracy_score,
        f1_score,
        log_loss,
        # _regression:
        mean_squared_error,
        # _ranking:
        roc_auc_score,
    )
except ImportError:
    print("sklearn not available")
