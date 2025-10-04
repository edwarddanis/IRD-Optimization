import pandas as pd
from emd_praxis.data.synthetic_generator import generate_dataset
from emd_praxis.ml.model import train_and_eval, batch_annotate

def test_train_and_eval_runs():
    df = generate_dataset(400, seed=0)
    model, res = train_and_eval(df, seed=0, n_estimators=20, learning_rate=0.1, max_depth=2, cv_folds=3)
    assert 0.0 <= res.auc <= 1.0
    analyzed = batch_annotate(df, model)
    assert "predicted_success_probability" in analyzed.columns
