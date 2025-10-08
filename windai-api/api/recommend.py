import pandas as pd  # imported
import numpy as np  # imported

def recommend_assemblies_from_predictions(df_original: pd.DataFrame,
                                          y_pred: np.ndarray,
                                          max_cost=350.0,
                                          max_u=1.2,
                                          target_shgc=0.45,
                                          tolerance=0.05,
                                          top_n=5) -> pd.DataFrame:  # defined
    results = df_original.copy()  # copied
    results['Predicted U-Factor'] = y_pred[:, 0]  # added
    results['Predicted SHGC'] = y_pred[:, 1]  # added
    results['Predicted Cost'] = y_pred[:, 2]  # added

    filtered = results[
        (results['Predicted Cost'] <= max_cost) &
        (results['Predicted U-Factor'] <= max_u) &
        (results['Predicted SHGC'].between(target_shgc - tolerance, target_shgc + tolerance))
    ].copy()  # filtered

    if filtered.empty:  # checked
        return filtered  # returned

    filtered['Score'] = (
        (1 / (1 + filtered['Predicted U-Factor'])) * 0.5 +
        (1 / (1 + filtered['Predicted Cost'])) * 0.5
    )  # scored

    top = filtered.sort_values(by='Score', ascending=False).head(top_n)  # sorted
    cols = [c for c in [
        'Glazing Name', 'Gas Fill Name', 'Spacer Name', 'Sealant Name',
        'Frame Name', 'Thermal Break Name',
        'Predicted U-Factor', 'Predicted SHGC', 'Predicted Cost'
    ] if c in top.columns]  # selected
    return top[cols].reset_index(drop=True)  # returned
