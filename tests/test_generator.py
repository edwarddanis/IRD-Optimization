import pandas as pd
from emd_praxis.data.synthetic_generator import generate_dataset

def test_generate_dataset_shapes():
    df = generate_dataset(200, seed=123)
    assert len(df) == 200
    assert {"project_name","tech_area","budget_musd","success"}.issubset(df.columns)
    assert df["budget_musd"].between(0.5,12.0).all()
    assert set(df["success"].unique()).issubset({0,1})
