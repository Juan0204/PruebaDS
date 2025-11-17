from pathlib import Path
import ast
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns            

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor


DIM_KEY = "Assembled Product Dimensions (L x W x H)"
WEI_KEY = "Assembled Product Weight"

EXTRA_COLS = [
    "root_category_name",
    "category_name",
    "brand",
    "sku",
    "product_name",
    "description",
    "review_tags",
    "categories",
    "sizes",
    "colors",
    "other_attributes",
    "unit",
]

CALC_COLS = [
    DIM_KEY,
    WEI_KEY,
    "length_in",
    "width_in",
    "height_in",
    "weight_lb",
]

NUM_RE = r"(\d+(?:[.,]\d+)?)"


def load_raw_data(csv_path: Path) -> pd.DataFrame:
    """
    Lee el CSV original y devuelve el DataFrame crudo.
    """
    df = pd.read_csv(csv_path)
    print(f"[LOAD] Filas cargadas: {len(df)}")
    return df


def parse_specifications(cell):
    """
    Devuelve un dict {name: value} a partir de 'specifications' en múltiples formatos:
    - list[{"name":..., "value":...}], dict plano, JSON, literal_eval, y texto "Clave → Valor".
    Si no se puede parsear, retorna {}.
    """
    if cell is None:
        return {}
    if isinstance(cell, dict):
        return cell
    if isinstance(cell, list):
        out = {}
        for d in cell:
            if isinstance(d, dict) and "name" in d and "value" in d:
                out[str(d["name"]).strip()] = d["value"]
        return out

    s = str(cell).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return {}

    # Intentar JSON o literal_eval
    for loader in (json.loads, ast.literal_eval):
        try:
            data = loader(s)
            if isinstance(data, list):
                out = {}
                for d in data:
                    if isinstance(d, dict) and "name" in d and "value" in d:
                        out[str(d["name"]).strip()] = d["value"]
                if out:
                    return out
            if isinstance(data, dict):
                return {str(k).strip(): v for k, v in data.items()}
        except Exception:
            pass

    # Fallback: "Clave → Valor"
    out = {}
    for line in s.splitlines():
        if "→" in line:
            k, v = line.split("→", 1)
            out[k.strip()] = v.strip()
    return out


def enrich_with_spec_dict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega la columna _spec_dict con el parseo de 'specifications'.
    """
    out = df.copy()
    out["_spec_dict"] = out["specifications"].apply(parse_specifications)
    return out


def _to_float(x):
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return None


def parse_weight_lb(text):
    """
    Convierte texto con unidades comunes a libras.
    Si la unidad está en kg, oz o g, convierte a libras.
    """
    if not text or (isinstance(text, float) and np.isnan(text)):
        return None
    t = str(text).lower()
    patterns = [
        ("lb", 1.0),
        ("lbs", 1.0),
        ("pound", 1.0),
        ("pounds", 1.0),
        ("kg", 2.20462),
        ("kilogram", 2.20462),
        ("kilograms", 2.20462),
        ("g", 1 / 453.592),
        ("gram", 1 / 453.592),
        ("grams", 1 / 453.592),
        ("oz", 1 / 16),
        ("ounce", 1 / 16),
        ("ounces", 1 / 16),
    ]
    for unit, factor in patterns:
        m = re.search(NUM_RE + r"\s*" + unit + r"\b", t)
        if m:
            val = _to_float(m.group(1))
            return val * factor if val is not None else None
    return None


def parse_dims_triplet_in(text):
    """
    Lee 'L x W x H' con unidad (Inches, in, cm, mm, m) y retorna (L_in, W_in, H_in) en pulgadas.
    """
    if not text or (isinstance(text, float) and np.isnan(text)):
        return (None, None, None)
    t = str(text).lower().replace("×", "x")
    m = re.search(
        NUM_RE + r"\s*x\s*" + NUM_RE + r"\s*x\s*" + NUM_RE + r"\s*([a-z\" ]+)$",
        t,
        flags=re.I,
    )
    if not m:
        return (None, None, None)
    a, b, c, unit = m.groups()
    a, b, c = _to_float(a), _to_float(b), _to_float(c)
    if None in (a, b, c):
        return (None, None, None)
    unit = unit.strip().replace('"', "")
    unit_map = {
        "inch": 1.0,
        "inches": 1.0,
        "in": 1.0,
        "cm": 1 / 2.54,
        "mm": 1 / 25.4,
        "m": 39.3701,
    }
    u = unit.split()[0] if unit else ""
    factor = unit_map.get(u, None)
    if factor is None:
        return (None, None, None)
    return (a * factor, b * factor, c * factor)


def add_physical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    A partir de _spec_dict crea columnas de dimensiones y peso numéricos.
    """
    out = df.copy()

    # Extraer texto de dimensiones y peso desde el dict
    out[DIM_KEY] = out["_spec_dict"].apply(
        lambda d: d.get(DIM_KEY) if isinstance(d, dict) else np.nan
    )
    out[WEI_KEY] = out["_spec_dict"].apply(
        lambda d: d.get(WEI_KEY) if isinstance(d, dict) else np.nan
    )

    # Parseo numérico
    out["weight_lb"] = out[WEI_KEY].apply(parse_weight_lb)
    out[["length_in", "width_in", "height_in"]] = out[DIM_KEY].apply(
        lambda s: pd.Series(parse_dims_triplet_in(s))
    )

    return out


def build_result_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye el DataFrame 'result' con las columnas de contexto + físicas.
    """
    extra_cols = [c for c in EXTRA_COLS if c in df.columns]
    all_cols = extra_cols + CALC_COLS
    all_cols = [c for c in all_cols if c in df.columns]
    result = df[all_cols].copy()
    print(f"[RESULT] Columns: {list(result.columns)}")
    print(f"[RESULT] Shape  : {result.shape}")
    return result


def enrich_extra_features(result: pd.DataFrame) -> pd.DataFrame:
    out = result.copy()

    list_cols = ["review_tags", "categories", "sizes", "colors", "other_attributes"]

    for col in list_cols:
        out[col + "_str"] = out[col].astype(str)
        out["n_" + col] = out[col].apply(lambda x: len(x) if isinstance(x, list) else 0)

    if "brand" in out.columns:
        out["brand_name"] = out["brand"]

    return out


def add_geom_feats(data: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega features geométricas derivadas de length_in, width_in y weight_lb.
    """
    out = data.copy()
    L = out.get("length_in")
    W = out.get("width_in")

    if L is None:
        L = pd.Series(np.nan, index=out.index)
    if W is None:
        W = pd.Series(np.nan, index=out.index)

    out["area_in2"] = L * W
    out["ratio_lw"] = L / (W + 1e-6)
    out["minLW"] = np.minimum(L, W)
    out["maxLW"] = np.maximum(L, W)
    out["geom_mean_lw"] = np.sqrt(np.clip(out["area_in2"], 0, None))  # √(L*W)
    out["diag_in"] = np.sqrt(np.clip(L**2 + W**2, 0, None))           # diagonal

    if "weight_lb" in out.columns:
        out["w_per_area"] = out["weight_lb"] / (out["area_in2"] + 1e-6)
        out["w_per_minLW"] = out["weight_lb"] / (out["minLW"] + 1e-6)
        out["w_per_maxLW"] = out["weight_lb"] / (out["maxLW"] + 1e-6)
        out["w_per_diag"] = out["weight_lb"] / (out["diag_in"] + 1e-6)

    for c in [
        "area_in2",
        "ratio_lw",
        "minLW",
        "maxLW",
        "geom_mean_lw",
        "diag_in",
        "w_per_area",
        "w_per_minLW",
        "w_per_maxLW",
        "w_per_diag",
    ]:
        if c in out.columns:
            out[f"log_{c}"] = np.log1p(out[c])

    return out


def make_ohe():
    """
    Crea el OneHotEncoder compatible con diferentes versiones de sklearn.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def eval_reg(
    data: pd.DataFrame,
    target_col: str,
    num_features,
    cat_features,
    use_log: bool = False,
    cv_splits: int = 5,
    test_size: float = 0.25,
):
    print("\n" + "=" * 70)
    print(f"[EVAL] Target: {target_col}")
    print(f"   num_features: {num_features}")
    print(f"   cat_features: {cat_features}")

    data = data[data[target_col].notna()].copy()
    print(f"   filas con etiqueta real: {len(data)}")
    if data.empty:
        print("   → No hay datos con etiqueta; no se puede evaluar este target.")
        return None

    X = data[num_features + cat_features].copy()
    y = data[target_col].astype(float).copy()

    preproc = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("ohe", make_ohe()),
                    ]
                ),
                cat_features,
            ),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    base_pipe = Pipeline(
        [
            ("prep", preproc),
            ("model", model),
        ]
    )

    if use_log:
        pipe = TransformedTargetRegressor(
            regressor=base_pipe,
            func=np.log1p,
            inverse_func=np.expm1,
        )
    else:
        pipe = base_pipe

    # Holdout
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print(f"   Holdout R²  : {r2_score(y_test, y_pred):.4f}")
    print(f"   Holdout MAE : {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"   Holdout RMSE: {rmse(y_test, y_pred):.4f}")

    # CV
    try:
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
        r2_scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
        mae_scores = -cross_val_score(
            pipe, X, y, cv=kf, scoring="neg_mean_absolute_error"
        )
        rm_scores = np.sqrt(
            -cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_squared_error")
        )
        print(
            f"   CV{cv_splits} R²   : {r2_scores.mean():.4f} ± {r2_scores.std():.4f}"
        )
        print(
            f"   CV{cv_splits} MAE  : {mae_scores.mean():.4f} ± {mae_scores.std():.4f}"
        )
        print(
            f"   CV{cv_splits} RMSE : {rm_scores.mean():.4f} ± {rm_scores.std():.4f}"
        )
    except Exception as e:
        print(f"   (CV no disponible): {e}")

    # Importancias por permutación
    try:
        pi = permutation_importance(
            pipe, X_test, y_test, n_repeats=8, random_state=42, n_jobs=-1
        )
        importances = pi.importances_mean
        order = np.argsort(importances)[::-1][:15]
        print("   Top importancias (permutation, sobre variables transformadas):")
        for i in order:
            print(f"     - feature_{i}: {importances[i]:.6f}")
    except Exception as e:
        print(f"   (Permutation importance no disponible): {e}")

    return pipe


def run_modeling_pipeline(result: pd.DataFrame):
    models = {}

    # Features categóricas (todas)
    context_cols = [
        c for c in [
            "root_category_name", "category_name", "brand_name", "product_name",
            "review_tags_str", "categories_str", "sizes_str",
            "colors_str", "unit", "other_attributes_str"
        ]
        if c in result.columns
    ]

    # Features numéricas (todas)
    extra_nums = [
        c for c in [
            "n_review_tags", "n_categories", "n_sizes",
            "n_colors", "n_other_attributes"
        ] if c in result.columns
    ]

    # LENGTH_IN
    num_feats_length = ["width_in", "height_in", "weight_lb"] + extra_nums
    models["length_in"] = eval_reg(
        result, "length_in", num_feats_length, context_cols, use_log=False
    )

    # WIDTH_IN
    num_feats_width = ["length_in", "height_in", "weight_lb"] + extra_nums
    models["width_in"] = eval_reg(
        result, "width_in", num_feats_width, context_cols, use_log=False
    )

    # HEIGHT_IN
    num_feats_height = ["length_in", "width_in", "weight_lb"] + extra_nums
    models["height_in"] = eval_reg(
        result, "height_in", num_feats_height, context_cols, use_log=True
    )

    # WEIGHT_LB
    num_feats_weight = ["length_in", "width_in", "height_in"] + extra_nums
    models["weight_lb"] = eval_reg(
        result, "weight_lb", num_feats_weight, context_cols, use_log=True
    )

    return models


def impute_missing_values(result: pd.DataFrame, models: dict) -> pd.DataFrame:
    """
    Usa los modelos entrenados para imputar valores faltantes de
    length_in, width_in, height_in, weight_lb y crea columnas *_final.
    """
    out = result.copy()
    context_cols = [
        c for c in [
            "root_category_name",
            "category_name",
            "brand_name",
            "product_name",
            "review_tags_str",
            "categories_str",
            "sizes_str",
            "colors_str",
            "unit",
            "other_attributes_str",
        ]
        if c in out.columns
    ]

    extra_num_feats = [
        c
        for c in [
            "n_review_tags",
            "n_categories",
            "n_sizes",
            "n_colors",
            "n_other_attributes",
        ]
        if c in out.columns
    ]

    #length_in
    num_feats_length = [
        c for c in ["width_in", "height_in", "weight_lb"] if c in out.columns
    ] + extra_num_feats

    if "length_in" in out.columns and models.get("length_in") is not None:
        mask_len = out["length_in"].isna()
        if mask_len.any():
            X_pred = out.loc[mask_len, num_feats_length + context_cols]
            out.loc[mask_len, "length_in"] = models["length_in"].predict(X_pred)

    #width_in
    num_feats_width = [
        c for c in ["length_in", "height_in", "weight_lb"] if c in out.columns
    ] + extra_num_feats

    if "width_in" in out.columns and models.get("width_in") is not None:
        mask_w = out["width_in"].isna()
        if mask_w.any():
            X_pred = out.loc[mask_w, num_feats_width + context_cols]
            out.loc[mask_w, "width_in"] = models["width_in"].predict(X_pred)

    #height_in
    num_feats_height = [
        c for c in ["length_in", "width_in", "weight_lb"] if c in out.columns
    ] + extra_num_feats

    if "height_in" in out.columns and models.get("height_in") is not None:
        mask_h = out["height_in"].isna()
        if mask_h.any():
            X_pred = out.loc[mask_h, num_feats_height + context_cols]
            out.loc[mask_h, "height_in"] = models["height_in"].predict(X_pred)

    #weight_lb
    num_feats_weight = [
        c for c in ["length_in", "width_in", "height_in"] if c in out.columns
    ] + extra_num_feats

    if "weight_lb" in out.columns and models.get("weight_lb") is not None:
        mask_we = out["weight_lb"].isna()
        if mask_we.any():
            X_pred = out.loc[mask_we, num_feats_weight + context_cols]
            out.loc[mask_we, "weight_lb"] = models["weight_lb"].predict(X_pred)

    out["length_in_final"] = out["length_in"]
    out["width_in_final"] = out["width_in"]
    out["height_in_final"] = out["height_in"]
    out["weight_lb_final"] = out["weight_lb"]

    return out


# PIPELINE PRINCIPAL (main)
def build_full_pipeline(csv_path: Path) -> tuple[pd.DataFrame, dict]:
    """
    Ejecuta todo el pipeline:
    1) Carga datos
    2) Parseo de specifications
    3) Cálculo de columnas físicas
    4) Vista result
    5) Features geométricas
    6) Modelado
    7) Imputación de faltantes

    Devuelve:
        - result_final: DataFrame final con todas las features e imputaciones
        - models: dict de modelos entrenados por target
    """
    # 1) Carga
    df_raw = load_raw_data(csv_path)

    # 2) Parseo specifications
    df_specs = enrich_with_spec_dict(df_raw)

    # 3) Columnas físicas numéricas
    df_phys = add_physical_columns(df_specs)
    
    # 4) Vista base result
    result = build_result_view(df_phys)

    result = enrich_extra_features(result)
    # 5) Features geométricas
    result_with_geom = add_geom_feats(result)

    # 6) Modelado
    models = run_modeling_pipeline(result_with_geom)

    # 7) Imputación de faltantes con los modelos entrenados
    result_imputed = impute_missing_values(result_with_geom, models)

    return result_imputed, models


def main(csv_path: Path):
    result_final, models = build_full_pipeline(csv_path)
    print("\n[PIPELINE] Completado.")
    print(f"[PIPELINE] result_final shape: {result_final.shape}")
    print(f"[PIPELINE] Modelos entrenados: {list(models.keys())}")
    return result_final, models

#if __name__ == "__main__":
#    main()
