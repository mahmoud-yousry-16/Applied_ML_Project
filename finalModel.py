import streamlit as st
import numpy as np
import os
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Air Quality Clustering", layout="wide")
st.title("Air Quality Data — Clustering Analysis")

# ─── Sidebar Navigation ───────────────────────────────────────────────────────
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to section:",
    [
        "1. Load Data",
        "2. Exploratory Analysis",
        "3. Preprocessing",
        "4. Feature Engineering",
        "5. KMeans Clustering",
        "6. DBSCAN Clustering",
    ],
)

# ─── Session state helpers ────────────────────────────────────────────────────
def get(key, default=None):
    return st.session_state.get(key, default)

def put(key, val):
    st.session_state[key] = val

# ─── Helper: prerequisite banner ─────────────────────────────────────────────
def need(key, msg):
    if get(key) is None:
        st.error(f"!!! {msg}")
        st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Load Data
# ══════════════════════════════════════════════════════════════════════════════
if section == "1. Load Data":
    st.header("Load Data")

    st.subheader("Option A — Download from Kaggle")
    col1, col2 = st.columns(2)
    with col1:
        kg_user = st.text_input("Kaggle Username", value="malakkhaledaboahmed")
    with col2:
        kg_key = st.text_input("Kaggle API Key", type="password")

    if st.button("Download & Load from Kaggle"):
        if not kg_user or not kg_key:
            st.error("Please enter both username and API key.")
        else:
            with st.spinner("Downloading dataset…"):
                os.environ["KAGGLE_USERNAME"] = kg_user
                os.environ["KAGGLE_KEY"] = kg_key
                os.system("kaggle datasets download -d fedesoriano/air-quality-data-set")
                with zipfile.ZipFile("air-quality-data-set.zip", "r") as z:
                    z.extractall("air_quality_data")
                df = pd.read_csv("air_quality_data/AirQuality.csv", sep=";", decimal=",")
                df.drop(df.columns[-2:], axis=1, inplace=True)
                df = df.replace(-200, np.nan)
                put("df_raw", df)
            st.success("Dataset loaded successfully.")
            st.dataframe(df.head(10))

    st.divider()

    st.subheader("Option B — Upload CSV manually")
    uploaded = st.file_uploader("Upload AirQuality.csv", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded, sep=";", decimal=",")
        df.drop(df.columns[-2:], axis=1, inplace=True)
        df = df.replace(-200, np.nan)
        put("df_raw", df)
        st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns.")
        st.dataframe(df.head(10))

    if get("df_raw") is not None:
        st.info(f"Dataset in memory: {get('df_raw').shape[0]} rows × {get('df_raw').shape[1]} columns.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Exploratory Analysis  (auto-runs on entry)
# ══════════════════════════════════════════════════════════════════════════════
elif section == "2. Exploratory Analysis":
    st.header("Exploratory Data Analysis")
    need("df_raw", "Please load the data first from Section 1.")

    df = get("df_raw")

    # ── Basic Info ────────────────────────────────────────────────────────────
    st.subheader("Basic Info")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Duplicates", df.duplicated().sum())
    st.dataframe(df.describe())

    # ── Missing Values ────────────────────────────────────────────────────────
    st.subheader("Missing Values")
    with st.spinner("Plotting missing values…"):
        fig, ax = plt.subplots(figsize=(12, 4))
        msno.bar(df, ax=ax)
        ax.set_title("Missing Values per Column")
        st.pyplot(fig)
        plt.close()

    # ── Outlier Check ─────────────────────────────────────────────────────────
    st.subheader("Outlier Check — Key Columns")
    cols_to_check = ["PT08.S3(NOx)", "NOx(GT)", "PT08.S1(CO)"]
    valid_cols = [c for c in cols_to_check if c in df.columns]
    if valid_cols:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(data=df[valid_cols], ax=ax)
        ax.set_title("Boxplot — Checking for Outliers")
        st.pyplot(fig)
        plt.close()

    # ── Rush Hour CO ──────────────────────────────────────────────────────────
    st.subheader("Rush Hour CO Analysis")
    df_tmp = df.copy()
    if "Date" in df_tmp.columns and "Time" in df_tmp.columns:
        df_tmp["Timestamp"] = pd.to_datetime(
            df_tmp["Date"] + " " + df_tmp["Time"].astype(str).str.replace(".", ":"),
            errors="coerce",
        )
        df_tmp["Hour"] = df_tmp["Timestamp"].dt.hour
        if "CO(GT)" in df_tmp.columns:
            fig, ax = plt.subplots(figsize=(12, 4))
            sns.lineplot(data=df_tmp, x="Hour", y="CO(GT)", marker="o", color="tab:blue", ax=ax)
            ax.set_title("Average CO(GT) Throughout the Day")
            ax.set_xticks(range(0, 24))
            ax.grid(True, linestyle="--", alpha=0.6)
            st.pyplot(fig)
            plt.close()

    # ── NOx by Day & Benzene Seasonal ─────────────────────────────────────────
    st.subheader("NOx by Day of Week  &  Benzene Seasonal Trend")
    if "Timestamp" in df_tmp.columns:
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        df_tmp["Day_of_Week"] = pd.Categorical(
            df_tmp["Timestamp"].dt.day_name(), categories=days_order, ordered=True
        )
        df_tmp["Month"] = df_tmp["Timestamp"].dt.month

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        if "NOx(GT)" in df_tmp.columns:
            sns.barplot(data=df_tmp, x="Day_of_Week", y="NOx(GT)", ax=axes[0], palette="coolwarm")
            axes[0].set_title("Average NOx(GT) by Day of Week")
            axes[0].tick_params(axis="x", rotation=45)
        if "C6H6(GT)" in df_tmp.columns:
            sns.lineplot(data=df_tmp, x="Month", y="C6H6(GT)", marker="s", ax=axes[1], color="red")
            axes[1].set_title("Seasonal Trend of Benzene (C6H6)")
            axes[1].set_xticks(range(1, 13))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Preprocessing  (auto-runs on entry)
# ══════════════════════════════════════════════════════════════════════════════
elif section == "3. Preprocessing":
    st.header("Preprocessing")
    need("df_raw", "Please load the data first from Section 1.")

    # Run once and cache; re-run if raw data changed
    raw_hash = id(get("df_raw"))
    if get("df_clean") is None or get("_preproc_hash") != raw_hash:
        with st.spinner("Running preprocessing pipeline…"):
            df = get("df_raw").copy()

            before = len(df)
            df = df.drop_duplicates()
            dropped_dup = before - len(df)

            for col in ["NMHC(GT)", "Date", "Time"]:
                if col in df.columns:
                    df = df.drop(col, axis=1)

            df = df.ffill().bfill()

            df_raw = get("df_raw").copy()
            if "Date" in df_raw.columns and "Time" in df_raw.columns:
                df["Timestamp"] = pd.to_datetime(
                    df_raw["Date"] + " " + df_raw["Time"].astype(str).str.replace(".", ":"),
                    errors="coerce",
                )
            if "Timestamp" not in df.columns:
                st.error("Timestamp column could not be created.")
                st.stop()

            df["Hour"] = df["Timestamp"].dt.hour
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            df["Day_of_Week"] = pd.Categorical(
                df["Timestamp"].dt.day_name(), categories=days_order, ordered=True
            )
            df["Month"] = df["Timestamp"].dt.month

            df = df[
                ~(
                    (df["Timestamp"].dt.date == pd.to_datetime("2004-11-03").date())
                    & (df["Timestamp"].dt.hour <= 7)
                )
            ]

            if "CO(GT)" in df.columns:
                Q1, Q99 = df["CO(GT)"].quantile(0.05), df["CO(GT)"].quantile(0.95)
                df["CO(GT)"] = df["CO(GT)"].clip(lower=Q1, upper=Q99)

            if "PT08.S5(O3)" in df.columns:
                Q1, Q99 = df["PT08.S5(O3)"].quantile(0.01), df["PT08.S5(O3)"].quantile(0.99)
                df["PT08.S5(O3)"] = df["PT08.S5(O3)"].clip(lower=Q1, upper=Q99)

            put("df_clean", df)
            put("_preproc_hash", raw_hash)
            put("dropped_dup", dropped_dup)

    df = get("df_clean")
    st.success("Preprocessing complete.")
    st.write(f"Duplicates removed: **{get('dropped_dup')}**")

    col1, col2 = st.columns(2)
    col1.metric("Rows after cleaning", df.shape[0])
    col2.metric("Columns", df.shape[1])

    st.subheader("Missing Values After Preprocessing")
    fig, ax = plt.subplots(figsize=(12, 4))
    msno.bar(df, ax=ax)
    ax.set_title("Missing Values After Preprocessing")
    st.pyplot(fig)
    plt.close()

    st.subheader("Correlation Heatmap (Post-Cleaning)")
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
    plt.close()

    st.subheader("Distribution of Key Variables")
    cols_to_plot = [c for c in ["CO(GT)", "C6H6(GT)", "T", "RH"] if c in df.columns]
    if cols_to_plot:
        fig, axes = plt.subplots(1, len(cols_to_plot), figsize=(5 * len(cols_to_plot), 4))
        if len(cols_to_plot) == 1:
            axes = [axes]
        colors = ["blue", "green", "orange", "red"]
        for i, col in enumerate(cols_to_plot):
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color=colors[i % len(colors)])
            axes[i].set_title(f"{col} Distribution")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Feature Engineering  (auto-runs on entry)
# ══════════════════════════════════════════════════════════════════════════════
elif section == "4. Feature Engineering":
    st.header("Feature Engineering")
    need("df_clean", "Please run Preprocessing first (Section 3).")

    clean_hash = id(get("df_clean"))
    if get("X_final") is None or get("_feat_hash") != clean_hash:
        with st.spinner("Engineering features…"):
            df = get("df_clean").copy()

            df["Hour"] = df["Timestamp"].dt.hour
            df["Is_RushHour"] = df["Hour"].apply(lambda x: 1 if x in [7, 8, 9, 17, 18, 19] else 0)
            df["Is_WorkingDay"] = df["Timestamp"].dt.dayofweek.apply(lambda x: 0 if x >= 5 else 1)

            def get_season(month):
                if month in [12, 1, 2]: return "Winter"
                elif month in [3, 4, 5]: return "Spring"
                elif month in [6, 7, 8]: return "Summer"
                else: return "Autumn"

            df["Season"] = df["Month"].apply(get_season)
            season_dummies = pd.get_dummies(df["Season"], prefix="Season").astype(int)
            df = pd.concat([df, season_dummies], axis=1)

            df["CO_lag1"] = df["CO(GT)"].shift(1)
            df["T_lag1"] = df["T"].shift(1)
            df["NOx_lag1"] = df["NOx(GT)"].shift(1)
            df["CO_mean_3h"] = df["CO(GT)"].rolling(window=3).mean()
            df["S1_std_6h"] = df["PT08.S1(CO)"].rolling(window=6).std()
            df["Temp_Diff"] = df["T"].diff().fillna(0)
            df["T_RH_Interaction"] = df["T"] * df["RH"]

            sensor_cols = [c for c in ["PT08.S1(CO)", "C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)", "NO2(GT)"] if c in df.columns]
            pca_var = None
            if len(sensor_cols) >= 2:
                scaler = StandardScaler()
                sensors_scaled = scaler.fit_transform(df[sensor_cols].fillna(0))
                pca = PCA(n_components=2)
                sensors_pca = pca.fit_transform(sensors_scaled)
                df["Sensor_PCA1"] = sensors_pca[:, 0]
                df["Sensor_PCA2"] = sensors_pca[:, 1]
                pca_var = pca.explained_variance_ratio_.round(3).tolist()

            df["CO_Trend"] = df["CO(GT)"].diff()

            def get_day_period(h):
                if 6 <= h <= 10: return 1
                if 16 <= h <= 20: return 2
                if 21 <= h or h <= 5: return 3
                return 0

            df["Day_Period"] = df["Hour"].apply(get_day_period)
            df["CO_Volatility_3h"] = df["CO(GT)"].rolling(window=3).std()
            df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

            final_feature_columns = [
                "CO_mean_3h", "CO_Trend", "CO_Volatility_3h", "CO_lag1",
                "Sensor_PCA1", "Sensor_PCA2",
                "T_RH_Interaction",
                "PT08.S2(NMHC)", "C6H6(GT)", "PT08.S4(NO2)", "NOx(GT)",
            ]
            final_feature_columns = [c for c in final_feature_columns if c in df.columns]

            # Drop rows with any NaN in feature columns
            df_final_clean = df.dropna(subset=final_feature_columns).reset_index(drop=True)

            features_to_scale = [c for c in final_feature_columns if c not in ["hour_sin", "hour_cos"]]
            circular_features = [c for c in ["hour_sin", "hour_cos"] if c in df_final_clean.columns]

            robust_scaler = RobustScaler()
            scaled_data = robust_scaler.fit_transform(df_final_clean[features_to_scale])
            X_scaled_part = pd.DataFrame(scaled_data, columns=features_to_scale)

            if circular_features:
                X_final = pd.concat([X_scaled_part, df_final_clean[circular_features].reset_index(drop=True)], axis=1)
            else:
                X_final = X_scaled_part

            # ── Final safety: guarantee zero NaNs before any sklearn call ──
            X_final = X_final.dropna()
            df_final_clean = df_final_clean.loc[X_final.index].reset_index(drop=True)
            X_final = X_final.reset_index(drop=True)

            put("df_final_clean", df_final_clean)
            put("X_final", X_final)
            put("final_feature_columns", final_feature_columns)
            put("_feat_hash", clean_hash)
            put("_pca_var", pca_var)

    st.success("✅ Feature engineering complete.")
    if get("_pca_var"):
        st.write(f"PCA explained variance: **{get('_pca_var')}**")
    st.write(f"Features used: `{get('final_feature_columns')}`")
    st.write(f"Dataset shape ready for clustering: **{get('X_final').shape}**")
    st.dataframe(get("X_final").head())

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — KMeans Clustering  (auto-runs; slider drives re-clustering)
# ══════════════════════════════════════════════════════════════════════════════
elif section == "5. KMeans Clustering":
    st.header("KMeans Clustering")
    need("X_final", "Please run Feature Engineering first (Section 4).")

    X_final = get("X_final")
    df_final_clean = get("df_final_clean")

    # ── Elbow Method (auto) ───────────────────────────────────────────────────
    st.subheader("Elbow Method")
    k_max = st.slider("Max K to evaluate", min_value=5, max_value=15, value=10)

    # Safety: ensure no NaN reaches sklearn
    X_final = X_final.dropna().reset_index(drop=True)

    elbow_key = f"elbow_{k_max}"
    if get(elbow_key) is None:
        with st.spinner("Computing inertia for each K…"):
            inertia = []
            K_range = range(1, k_max + 1)
            for k in K_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(X_final)
                inertia.append(km.inertia_)
            put(elbow_key, (list(K_range), inertia))

    K_range, inertia = get(elbow_key)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(K_range, inertia, marker="o", linestyle="--", color="b")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method for Air Quality Clustering")
    ax.set_xticks(K_range)
    ax.grid(True)
    st.pyplot(fig)
    plt.close()

    # ── Apply KMeans (auto, re-runs when slider changes) ──────────────────────
    st.subheader("KMeans Results")
    n_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=3)

    km_key = f"kmeans_{n_clusters}"
    if get(km_key) is None:
        with st.spinner(f"Clustering with K={n_clusters}…"):
            kmeans_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans_final.fit_predict(X_final)
            df_km = df_final_clean.copy()
            df_km["Cluster"] = clusters
            score = silhouette_score(X_final, clusters)
            put(km_key, (df_km, score))

    df_km, score = get(km_key)
    st.metric("Silhouette Score", f"{score:.3f}")

    st.subheader("Points per Cluster")
    st.dataframe(df_km["Cluster"].value_counts().rename("Count").to_frame())

    st.subheader("Cluster Analysis (Mean Values)")
    agg_cols = {c: "mean" for c in ["CO(GT)", "NOx(GT)", "T", "Hour", "Is_RushHour"] if c in df_km.columns}
    st.dataframe(df_km.groupby("Cluster").agg(agg_cols).sort_values("CO(GT)").round(3))

    st.subheader("Cluster Visualization (PCA Space)")
    if "Sensor_PCA1" in df_km.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(
            df_km["Sensor_PCA1"], df_km["Sensor_PCA2"],
            c=df_km["Cluster"], cmap="Set1", s=10, alpha=0.6,
        )
        plt.colorbar(scatter, ax=ax, label="Cluster")
        ax.set_xlabel("Sensor PCA1")
        ax.set_ylabel("Sensor PCA2")
        ax.set_title(f"KMeans Clusters (K={n_clusters})")
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — DBSCAN Clustering  (auto-runs; sliders drive re-run)
# ══════════════════════════════════════════════════════════════════════════════
elif section == "6. DBSCAN Clustering":
    st.header("DBSCAN Clustering")
    need("X_final", "Please run Feature Engineering first (Section 4).")

    X_final = get("X_final").dropna().reset_index(drop=True)
    df_final_clean = get("df_final_clean")

    # ── K-Distance Graph (auto) ───────────────────────────────────────────────
    st.subheader("K-Distance Graph")
    min_samples_val = st.slider("min_samples", min_value=5, max_value=50, value=22)

    kdist_key = f"kdist_{min_samples_val}"
    if get(kdist_key) is None:
        with st.spinner("Computing K-distance graph…"):
            nn = NearestNeighbors(n_neighbors=min_samples_val)
            nn.fit(X_final)
            distances, _ = nn.kneighbors(X_final)
            k_distances = np.sort(distances[:, -1])
            put(kdist_key, k_distances)

    k_distances = get(kdist_key)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(k_distances)
    ax.axhline(y=1.8, color="r", linestyle="--", label="Suggested Start EPS")
    ax.axhline(y=2.2, color="g", linestyle="--", label="Suggested End EPS")
    ax.set_title(f"K-Distance Graph (n_neighbors={min_samples_val})")
    ax.legend()
    st.pyplot(fig)
    plt.close()

    # ── EPS Search (auto) ─────────────────────────────────────────────────────
    st.subheader("EPS Search")
    col1, col2, col3 = st.columns(3)
    with col1:
        eps_start = st.number_input("EPS search start", value=1.0, step=0.1)
    with col2:
        eps_end = st.number_input("EPS search end", value=2.6, step=0.1)
    with col3:
        eps_step = st.number_input("EPS step", value=0.1, step=0.05)

    search_key = f"dbscan_search_{min_samples_val}_{eps_start}_{eps_end}_{eps_step}"
    if get(search_key) is None:
        with st.spinner("Searching for optimal EPS…"):
            eps_range = np.arange(eps_start, eps_end, eps_step)
            scores, noise_ratios = [], []
            for eps in eps_range:
                db = DBSCAN(eps=eps, min_samples=min_samples_val).fit(X_final)
                labels = db.labels_
                mask = labels != -1
                n_cls = len(set(labels[mask]))
                scores.append(silhouette_score(X_final[mask], labels[mask]) if n_cls > 1 else -1)
                noise_ratios.append(np.sum(labels == -1) / len(labels))

            optimal_eps = eps_range[np.argmax(scores)]
            best_score = max(scores)
            put(search_key, (eps_range, scores, optimal_eps, best_score))

    eps_range, scores, optimal_eps, best_score = get(search_key)

    st.success(f"Optimal EPS: **{optimal_eps:.2f}** | Silhouette: **{best_score:.4f}**")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eps_range, scores, marker="o", color="blue", label="Silhouette Score")
    ax.axvline(x=optimal_eps, color="red", linestyle="--", label=f"Best EPS = {optimal_eps:.2f}")
    ax.set_xlabel("EPS")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("EPS vs Silhouette Score")
    ax.legend()
    st.pyplot(fig)
    plt.close()

    # ── Final DBSCAN Model (auto) ─────────────────────────────────────────────
    st.subheader("Final DBSCAN Results")
    st.write(f"Using EPS = **{optimal_eps:.2f}**, min_samples = **{min_samples_val}**")

    final_key = f"dbscan_final_{min_samples_val}_{optimal_eps:.2f}"
    if get(final_key) is None:
        with st.spinner("Fitting final DBSCAN model…"):
            best_dbscan = DBSCAN(eps=optimal_eps, min_samples=min_samples_val)
            db_labels = best_dbscan.fit_predict(X_final)
            df_db = df_final_clean.copy()
            df_db["DBSCAN_Cluster"] = db_labels
            n_cls = len(set(db_labels)) - (1 if -1 in db_labels else 0)
            noise_pct = np.sum(db_labels == -1) / len(db_labels) * 100
            put(final_key, (df_db, n_cls, noise_pct))

    df_db, n_cls, noise_pct = get(final_key)

    col1, col2 = st.columns(2)
    col1.metric("Clusters Found", n_cls)
    col2.metric("Noise %", f"{noise_pct:.2f}%")

    if "Sensor_PCA1" in df_db.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        unique_labels = sorted(df_db["DBSCAN_Cluster"].unique())
        colors = plt.cm.get_cmap("tab10", len(unique_labels))
        for i, label in enumerate(unique_labels):
            mask = df_db["DBSCAN_Cluster"] == label
            ax.scatter(
                df_db.loc[mask, "Sensor_PCA1"],
                df_db.loc[mask, "Sensor_PCA2"],
                c=[colors(i)],
                marker="x" if label == -1 else "o",
                s=15, alpha=0.6,
                label="Noise" if label == -1 else f"Cluster {label}",
            )
        ax.legend(markerscale=2)
        ax.set_xlabel("Sensor PCA1")
        ax.set_ylabel("Sensor PCA2")
        ax.set_title(f"Final DBSCAN (eps={optimal_eps:.2f})")
        st.pyplot(fig)
        plt.close()

    st.subheader("Cluster Mean Values")
    agg_cols = {c: "mean" for c in ["CO(GT)", "NOx(GT)", "T", "Is_RushHour"] if c in df_db.columns}
    st.dataframe(df_db.groupby("DBSCAN_Cluster").agg(agg_cols).round(3))