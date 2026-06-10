import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "online_shoppers_intention.csv"


def load_experiment_data(seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # Primary metric: Revenue = purchase completed
    df["converted"] = df["Revenue"].astype(int)

    # Guardrail proxies — all naturally present in the dataset
    # session_duration: total time spent across page categories
    df["session_duration"] = (
        df["Administrative_Duration"].fillna(0)
        + df["Informational_Duration"].fillna(0)
        + df["ProductRelated_Duration"].fillna(0)
    )
    # bounce_rate and page_values are direct columns
    df["bounce_rate"] = df["BounceRates"]
    df["page_value"] = df["PageValues"]

    # Random A/B assignment (simulated experiment)
    rng = np.random.default_rng(seed)
    df["group"] = rng.choice(["control", "variant"], size=len(df))

    cols = [
        "group", "converted",
        "session_duration", "bounce_rate", "page_value",
        "VisitorType", "Month", "Weekend",
    ]
    return df[cols].reset_index(drop=True)
