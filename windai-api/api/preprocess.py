import re  # imported
import json  # imported
import pandas as pd  # imported
import numpy as np  # imported

# canonical column groups  # defined
TARGET_COLS = ['Total U-Factor (W/m²·K)', 'SHGC', 'Total Cost (USD/m²)']  # defined
CATEGORICAL_COLS = ['Glazing Name', 'Gas Fill Name', 'Spacer Name', 'Sealant Name', 'Frame Name', 'Thermal Break Name']  # defined
DROP_IF_PRESENT = ['Source / Notes', 'Source/Notes', 'Glazing Layers List']  # defined

# --- parsing helpers (kept from your logic, hardened) ---
def parse_glass_layers(text):  # defined
    if pd.isna(text):  # checked
        return []  # returned
    s = str(text).strip().replace('×', 'x')  # normalized
    if s in ('', 'None', 'none', '–'):  # checked
        return []  # returned
    layers = []  # initialized
    for segment in s.split('+'):  # iterated
        seg = segment.strip()  # stripped
        if not seg:  # checked
            continue  # continued
        if 'x' in seg:  # handled NxY
            parts = seg.split('x')  # split
            if len(parts) == 2:  # checked
                try:
                    count = int(float(parts[0].strip()))  # parsed
                except:
                    count = 1  # defaulted
                try:
                    thick = float(parts[1].strip())  # parsed
                except:
                    thick = float(re.sub(r'[^\d\.]', '', parts[1]) or 0)  # parsed
                layers.extend([thick] * count)  # extended
            else:
                nums = re.findall(r'[\d.]+', seg)  # matched
                if len(nums) == 2:  # checked
                    count = int(float(nums[0]))  # parsed
                    thick = float(nums[1])  # parsed
                    layers.extend([thick] * count)  # extended
        else:
            try:
                thick = float(seg)  # parsed
            except:
                thick = float(re.sub(r'[^\d\.]', '', seg) or 0)  # parsed
            if thick:  # checked
                layers.append(thick)  # appended
    return layers  # returned

def parse_value(s):  # defined
    if pd.isna(s):  # checked
        return np.nan  # returned
    if isinstance(s, (int, float)):  # checked
        return float(s)  # returned
    val = str(s).strip().replace('≈', '').replace('~', '')  # cleaned
    if val in ('', '–', 'None', 'none'):  # checked
        return np.nan  # returned
    # average ranges "a–b" or "a-b"  # commented
    if re.match(r'^[\d\.]+\s*[\u2013\-]\s*[\d\.]+$', val):  # checked
        a, b = re.split(r'[\u2013\-]', val)  # split
        try:
            return (float(a) + float(b)) / 2.0  # returned
        except:
            pass  # passed
    val = re.sub(r'\(.*?\)', '', val)  # removed units in parentheses
    val = val.replace('$', '').replace(',', '')  # cleaned
    val = re.sub(r'[^\d\.]+', '', val)  # stripped
    if val in ('', '.'):  # checked
        return np.nan  # returned
    try:
        return float(val)  # returned
    except:
        return np.nan  # returned

def parse_frame_lambda(s):  # defined
    if pd.isna(s):  # checked
        return np.nan  # returned
    val = str(s).replace('~', '').strip()  # cleaned
    if val in ('', '–', 'None', 'none'):  # checked
        return np.nan  # returned
    if '/' in val:  # handled base/break pair
        base = val.split('/')[0].strip()  # split
        try:
            return float(base)  # returned
        except:
            nums = re.findall(r'[\d.]+', base)  # matched
            return float(nums[0]) if nums else np.nan  # returned
    return parse_value(val)  # delegated

# --- top-level cleaner ---
def clean_and_parse(df: pd.DataFrame) -> pd.DataFrame:  # defined
    df = df.copy()  # copied

    # drop note columns if present  # commented
    for c in DROP_IF_PRESENT:  # iterated
        if c in df.columns:  # checked
            df = df.drop(columns=c)  # dropped

    # glazing layers derived vars  # commented
    if 'Glazing Thickness (mm)' in df.columns:  # checked
        df['Glazing Layers List'] = df['Glazing Thickness (mm)'].apply(parse_glass_layers)  # parsed
        df['Num Glazing Layers'] = df['Glazing Layers List'].apply(len)  # derived
        df['Total Glass Thickness (mm)'] = df['Glazing Layers List'].apply(lambda L: sum(L) if L else 0.0)  # summed
        df['Laminated Glass Present'] = df['Glazing Layers List'].apply(lambda L: 1 if any(abs(t - round(t)) > 1e-6 for t in L) else 0)  # flagged
        # replace text thickness with summed numeric for consistency  # commented
        df['Glazing Thickness (mm)'] = df['Total Glass Thickness (mm)']  # replaced
        df = df.drop(columns=['Glazing Layers List'])  # dropped

    # categorical normalizations  # commented
    if 'Glazing Name' in df.columns:  # checked
        df['Glazing Name'] = df['Glazing Name'].astype(str).str.strip()  # normalized
    if 'Gas Fill Name' in df.columns:  # checked
        df['Gas Fill Name'] = df['Gas Fill Name'].replace({
            'Air (default fill)': 'Air',
            'Air (evacuated)': 'Vacuum',
            'Argon Gas': 'Argon',
            'Krypton Gas': 'Krypton',
            'Krypton Gas (90%)': 'Krypton',
        }).astype(str).str.strip()  # normalized
    if 'Spacer Name' in df.columns:  # checked
        df['Spacer Name'] = df['Spacer Name'].fillna('None').astype(str).str.strip()  # filled
        df['Spacer Name'] = df['Spacer Name'].apply(lambda x: 'None' if x.lower().startswith('none') else x)  # unified
        df['Spacer Name'] = df['Spacer Name'].replace({
            'Aluminum Spacer': 'Aluminum',
            'Stainless Steel Spacer': 'Stainless Steel',
            'Composite Spacer (Fiberglass/Polymer)': 'Composite (Fiberglass/Polymer)',
            'Silicone Foam Spacer': 'Silicone Foam',
            'Thermoplastic Spacer (TPS)': 'Thermoplastic (TPS)',
            '“Warm-Edge” Spacer (Hybrid Polymer)': 'Warm-Edge (Hybrid Polymer)',
            'Warm-Edge Spacer (Hybrid Polymer)': 'Warm-Edge (Hybrid Polymer)',
            'uPVC Foam (Thermal Break)': 'uPVC Foam',
        })  # normalized
        df['Spacer Name'] = df['Spacer Name'].str.replace('  ', ' ')  # cleaned
    if 'Sealant Name' in df.columns:  # checked
        df['Sealant Name'] = df['Sealant Name'].replace({r'Warm-edge Silicone.*': 'Warm-edge Silicone'}, regex=True).astype(str).str.strip()  # normalized
    if 'Frame Name' in df.columns:  # checked
        s = df['Frame Name'].astype(str).fillna('').str.strip()  # prepared
        s = s.apply(lambda x: x.split('+')[0].strip() if '+' in x else x)  # split
        s = s.str.replace(r'\(no thermal break\)', '', case=False, regex=True).str.strip()  # cleaned
        s = s.str.replace('Frame', '', case=False, regex=False).str.strip()  # cleaned
        s = s.str.replace(r'\(.*\)', '', regex=True).str.strip()  # cleaned
        s = s.replace({
            'uPVC (unplasticized polyvinyl chloride)': 'uPVC',
            'Aluminum-Clad uPVC': 'Aluminum-Clad uPVC',
            'Aluminum-Clad Wood': 'Aluminum-Clad Wood',
            'Fiberglass (Pultruded FRP)': 'Fiberglass',
            'Wood (Pine, Softwood)': 'Wood',
            'Wood (Pine)': 'Wood'
        })  # normalized
        df['Frame Name'] = s  # assigned
    if 'Thermal Break Name' in df.columns:  # checked
        df['Thermal Break Name'] = df['Thermal Break Name'].fillna('None').astype(str).str.strip()  # filled
        df['Thermal Break Name'] = df['Thermal Break Name'].replace({
            'Polyamide 6.6 GF (Thermal Break)': 'Polyamide 6.6 GF',
            'uPVC Foam (Thermal Break)': 'uPVC Foam',
            'Aerogel-Insulated Thermal Break': 'Aerogel-Insulated'
        }).str.strip()  # normalized

    # numeric parsing  # commented
    for col in ['Glazing λ (W/m·K)', 'Gas λ (W/m·K)', 'Spacer λ (W/m·K)', 'Spacer Width (mm)',
                'Sealant λ (W/m·K)', 'Frame Thickness (mm)', 'Thermal Break λ (W/m·K)']:  # iterated
        if col in df.columns:  # checked
            df[col] = df[col].apply(parse_value)  # parsed

    if 'Frame λ (W/m·K)' in df.columns:  # checked
        # split "base / break" if found  # commented
        mask = df['Frame λ (W/m·K)'].astype(str).str.contains('/', regex=False)  # created
        if mask.any():  # checked
            for idx in df[mask].index:  # iterated
                parts = str(df.at[idx, 'Frame λ (W/m·K)']).split('/')  # split
                base_val = parse_value(parts[0])  # parsed
                df.at[idx, 'Frame λ (W/m·K)'] = base_val  # set
                if 'Thermal Break Name' in df.columns and str(df.at[idx, 'Thermal Break Name']).lower() not in ['none', 'nan']:  # checked
                    if 'Thermal Break λ (W/m·K)' in df.columns:  # checked
                        if len(parts) > 1:  # checked
                            df.at[idx, 'Thermal Break λ (W/m·K)'] = parse_value(parts[1])  # set
        df['Frame λ (W/m·K)'] = df['Frame λ (W/m·K)'].apply(parse_frame_lambda)  # parsed

    # targets numeric  # commented
    for t in [c for c in TARGET_COLS if c in df.columns]:  # iterated
        df[t] = df[t].apply(parse_value)  # parsed

    # presence flags  # commented
    if 'Spacer Name' in df.columns:  # checked
        df['Spacer Present'] = df['Spacer Name'].apply(lambda x: 0 if str(x).lower() == 'none' else 1)  # flagged
    if 'Thermal Break Name' in df.columns:  # checked
        df['Thermal Break Present'] = df['Thermal Break Name'].apply(lambda x: 0 if str(x).lower() == 'none' else 1)  # flagged

    # explicit -1 for absent numeric (your preference)  # commented
    if 'Spacer Present' in df.columns:  # checked
        for c in ['Spacer λ (W/m·K)', 'Spacer Width (mm)']:  # iterated
            if c in df.columns:  # checked
                df.loc[df['Spacer Present'] == 0, c] = -1  # assigned
    if 'Thermal Break Present' in df.columns and 'Thermal Break λ (W/m·K)' in df.columns:  # checked
        df.loc[df['Thermal Break Present'] == 0, 'Thermal Break λ (W/m·K)'] = -1  # assigned

    return df  # returned

def build_features_train(df: pd.DataFrame):
    dfc = clean_and_parse(df)  # cleaned
    X = dfc.drop(columns=[c for c in TARGET_COLS if c in dfc.columns], errors='ignore')  # split
    y = dfc[[c for c in TARGET_COLS if c in dfc.columns]].copy() if all(c in dfc.columns for c in TARGET_COLS) else None  # split
    # one-hot encode  # commented
    X = pd.get_dummies(X, columns=[c for c in CATEGORICAL_COLS if c in X.columns], drop_first=False)  # encoded
    feature_columns = X.columns.tolist()  # captured
    return X, y, feature_columns  # returned

def build_features_infer(df_raw: pd.DataFrame, feature_columns: list):
    dfc = clean_and_parse(df_raw)  # cleaned
    X_new = dfc.drop(columns=[c for c in TARGET_COLS if c in dfc.columns], errors='ignore')  # split
    X_new = pd.get_dummies(X_new, columns=[c for c in CATEGORICAL_COLS if c in X_new.columns], drop_first=False)  # encoded
    # align columns  # commented
    for col in feature_columns:  # iterated
        if col not in X_new.columns:  # checked
            X_new[col] = 0  # added
    extra = [c for c in X_new.columns if c not in feature_columns]  # collected
    if extra:  # checked
        X_new = X_new.drop(columns=extra)  # dropped
    X_new = X_new[feature_columns]  # reordered
    return X_new  # returned

def save_feature_columns(path: str, feature_columns: list):
    with open(path, 'w') as f:  # opened
        json.dump(feature_columns, f)  # saved

def load_feature_columns(path: str) -> list:
    with open(path, 'r') as f:  # opened
        return json.load(f)  # loaded
