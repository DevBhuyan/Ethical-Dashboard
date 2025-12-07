def preprocess_data(df: pd.DataFrame,
                    dataset_name: str,
                    one_hot_encode: bool = False) -> pd.DataFrame:
    df = df.copy()

    protected_col = "Class"   # <-- special handling rule

    # convert boolean columns safely, exclude Class
    bool_cols = df.select_dtypes(
        include='bool').columns.difference([protected_col])
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # categorical columns (object/category)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    category_maps = {}
    dummies_list = []
    scalers = {}

    for col in cat_cols:

        unique_count = df[col].nunique(dropna=True)

        # ----- Always label-encode CLASS, regardless of settings -----
        if col == protected_col:
            le = LabelEncoder()
            ser = df[col].astype(str).fillna("<<MISSING>>")
            df[col] = le.fit_transform(ser)

            classes = le.classes_
            encoded = le.transform(classes)
            mapping = dict(zip(encoded, classes))
            category_maps[col] = mapping
            continue

        # Skip large cardinality categorical variables
        if unique_count > 100:
            warn(
                f"Skipping LabelEncoding for column '{col}' in dataset '{dataset_name}' "
                f"because it has {unique_count} unique values (> 100)."
            )
            continue

        # ----- One-hot encoding only if flag enabled -----
        if one_hot_encode and unique_count <= 10:
            ser = df[col].astype(object).where(df[col].notna(), "<<MISSING>>")
            dummies = pd.get_dummies(ser, prefix=col, dtype=float)
            dummies_list.append(dummies)
            df.drop(columns=[col], inplace=True)
            continue

        # ----- Otherwise, label encode -----
        le = LabelEncoder()
        ser = df[col].astype(str).fillna("<<MISSING>>")
        df[col] = le.fit_transform(ser)

        classes = le.classes_
        encoded = le.transform(classes)
        mapping = dict(zip(encoded, classes))
        category_maps[col] = mapping

    # Add dummy columns if any
    if dummies_list:
        try:
            all_dummies = pd.concat(dummies_list, axis=1)
            all_dummies.index = df.index
            df = pd.concat([df, all_dummies], axis=1)
        except Exception:
            warn("Failed to concat dummy columns. Attempting per-column fallback.")
            for d in dummies_list:
                df = pd.concat([df, d], axis=1)

    # ----- FLOAT SCALING (EXCLUDE CLASS) -----
    float_cols = [
        c for c in df.select_dtypes(include=[np.floating]).columns
        if c != protected_col
    ]

    for col in float_cols:
        try:
            col_vals = df[col]
            col_min = col_vals.min(skipna=True)
            col_max = col_vals.max(skipna=True)

            if pd.isna(col_min) and pd.isna(col_max):
                scalers[col] = {"min": None, "max": None}
                continue

            if col_max == col_min:
                df[col] = col_vals.apply(
                    lambda x: np.nan if pd.isna(x) else 0.0)
                scalers[col] = {"min": float(col_min), "max": float(col_max)}
                continue

            df[col] = (col_vals - col_min) / (col_max - col_min)
            scalers[col] = {"min": float(col_min), "max": float(col_max)}

        except Exception:
            warn(
                f"Failed to scale float column '{col}' in dataset '{dataset_name}'.")
            traceback.print_exc()

    # ----- IMPUTATION (EXCLUDE CLASS) -----
    for col in df.columns:
        if col == protected_col:
            continue

        if pd.api.types.is_numeric_dtype(df[col].dtype):
            imputer = SimpleImputer(strategy='mean')
        else:
            imputer = SimpleImputer(strategy='most_frequent')

        try:
            df[col] = imputer.fit_transform(df[[col]]).ravel()
        except Exception:
            warn(f"Couldn't impute {col} column of {dataset_name}.")
            traceback.print_exc()

    print(
        f"Adding the following category mapping to session state for {dataset_name}: \n"
        f"{category_maps}\n"
    )

    if get_script_run_ctx():
        ss.category_maps[dataset_name] = category_maps

    return df
