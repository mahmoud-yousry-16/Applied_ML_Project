import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Air Quality Analysis",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Mono', monospace !important; }
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .main-title { font-family: 'Space Mono', monospace; font-size: 2.2rem; font-weight: 700; color: #58a6ff; letter-spacing: -1px; margin-bottom: 0; }
    .subtitle { color: #8b949e; font-size: 1rem; margin-top: 4px; margin-bottom: 24px; }
    .metric-card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 20px; text-align: center; }
    .metric-value { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; color: #58a6ff; }
    .metric-label { color: #8b949e; font-size: 0.85rem; margin-top: 4px; }
    .section-header { font-family: 'Space Mono', monospace; font-size: 1.1rem; color: #3fb950; border-left: 3px solid #3fb950; padding-left: 12px; margin: 24px 0 16px 0; }
    .info-box { background: #161b22; border: 1px solid #30363d; border-left: 3px solid #58a6ff; border-radius: 6px; padding: 14px 18px; margin: 12px 0; font-size: 0.9rem; color: #c9d1d9; }
    .warn-box { background: #1c1a10; border: 1px solid #30363d; border-left: 3px solid #d29922; border-radius: 6px; padding: 14px 18px; margin: 12px 0; font-size: 0.9rem; color: #c9d1d9; }
    .stButton > button { background: #238636; color: white; border: none; border-radius: 6px; font-family: 'Space Mono', monospace; font-size: 0.85rem; padding: 10px 20px; }
    div[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
    .stTabs [data-baseweb="tab-list"] { background: #161b22; border-bottom: 1px solid #30363d; gap: 4px; }
    .stTabs [data-baseweb="tab"] { font-family: 'Space Mono', monospace; font-size: 0.8rem; color: #8b949e; background: transparent; border: none; padding: 10px 18px; }
    .stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom: 2px solid #58a6ff !important; background: transparent !important; }
    .stDataFrame { border: 1px solid #30363d; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#161b22', 'axes.facecolor':   '#0d1117',
    'axes.edgecolor':   '#30363d', 'axes.labelcolor':  '#c9d1d9',
    'xtick.color':      '#8b949e', 'ytick.color':      '#8b949e',
    'text.color':       '#c9d1d9', 'grid.color':       '#21262d',
    'grid.alpha':       0.6,       'axes.titlecolor':  '#e6edf3',
    'axes.titlesize':   12,        'axes.titleweight': 'bold',
})

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# الداتا بتتحفظ هنا مرة واحدة وبتتنقل لكل الصفحات تلقائياً
# ══════════════════════════════════════════════════════════════════════════════
for key in ['df_raw', 'df_before_clean', 'df_after_clean', 'df_final_clean',
            'X_final', 'feat_cols', 'loaded_file_name']:
    if key not in st.session_state:
        st.session_state[key] = None

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# محفوظة بـ cache — بتشتغل مرة واحدة بس لنفس الفايل
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_and_preprocess(file_bytes):
    import io
    df = pd.read_csv(io.BytesIO(file_bytes), sep=';', decimal=',')
    df.drop(df.columns[-2:], axis=1, inplace=True)

    df_raw = df.copy()
    df = df.replace(-200, np.nan)

    df['Timestamp'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'].astype(str).str.replace('.', ':'),
        errors='coerce'
    )
    df['Hour']        = df['Timestamp'].dt.hour
    df['Day_of_Week'] = df['Timestamp'].dt.day_name()
    df['Month']       = df['Timestamp'].dt.month
    days_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    df['Day_of_Week'] = pd.Categorical(df['Day_of_Week'], categories=days_order, ordered=True)
    df['Time'] = df['Timestamp'].dt.time
    df_before_clean = df.copy()

    # Preprocessing
    df = df.drop_duplicates()
    df = df.drop(['NMHC(GT)', 'Date', 'Time'], axis=1)
    df = df.ffill().bfill()
    df = df[~(
        (df['Timestamp'].dt.date == pd.to_datetime('2004-11-03').date()) &
        (df['Timestamp'].dt.hour <= 7)
    )]
    for col, lo, hi in [('CO(GT)', 0.05, 0.95), ('PT08.S5(O3)', 0.01, 0.99)]:
        df[col] = df[col].clip(lower=df[col].quantile(lo), upper=df[col].quantile(hi))
    df_after_clean = df.copy()

    # Feature Engineering
    df['Hour']          = df['Timestamp'].dt.hour
    df['Is_RushHour']   = df['Hour'].apply(lambda x: 1 if x in [7,8,9,17,18,19] else 0)
    df['Is_WorkingDay'] = df['Timestamp'].dt.dayofweek.apply(lambda x: 0 if x >= 5 else 1)

    def get_season(month):
        if month in [12,1,2]:  return 'Winter'
        elif month in [3,4,5]: return 'Spring'
        elif month in [6,7,8]: return 'Summer'
        else:                  return 'Autumn'
    df['Season'] = df['Month'].apply(get_season)
    df = pd.concat([df, pd.get_dummies(df['Season'], prefix='Season').astype(int)], axis=1)

    df['CO_lag1']          = df['CO(GT)'].shift(1)
    df['T_lag1']           = df['T'].shift(1)
    df['NOx_lag1']         = df['NOx(GT)'].shift(1)
    df['CO_mean_3h']       = df['CO(GT)'].rolling(3).mean()
    df['S1_std_6h']        = df['PT08.S1(CO)'].rolling(6).std()
    df['Temp_Diff']        = df['T'].diff().fillna(0)
    df['T_RH_Interaction'] = df['T'] * df['RH']

    sensor_cols    = ['PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'NO2(GT)']
    sensors_scaled = StandardScaler().fit_transform(df[sensor_cols])
    sensors_pca    = PCA(n_components=2).fit_transform(sensors_scaled)
    df['Sensor_PCA1'] = sensors_pca[:, 0]
    df['Sensor_PCA2'] = sensors_pca[:, 1]

    df['CO_Trend']         = df['CO(GT)'].diff()
    df['CO_Volatility_3h'] = df['CO(GT)'].rolling(3).std()
    df['hour_sin']         = np.sin(2 * np.pi * df['Hour'] / 24)
    df['hour_cos']         = np.cos(2 * np.pi * df['Hour'] / 24)

    def get_day_period(h):
        if 6 <= h <= 10:      return 1
        if 16 <= h <= 20:     return 2
        if 21 <= h or h <= 5: return 3
        return 0
    df['Day_Period'] = df['Hour'].apply(get_day_period)

    final_feature_columns = [
        'CO_mean_3h','CO_Trend','CO_Volatility_3h','CO_lag1',
        'Sensor_PCA1','Sensor_PCA2','T_RH_Interaction',
        'PT08.S2(NMHC)','C6H6(GT)','PT08.S4(NO2)','NOx(GT)'
    ]
    df_final_clean = df.dropna(subset=final_feature_columns).copy()
    features_to_scale = [c for c in final_feature_columns if c not in ['hour_sin','hour_cos']]
    scaled_data   = RobustScaler().fit_transform(df_final_clean[features_to_scale])
    X_scaled_part = pd.DataFrame(scaled_data, columns=features_to_scale, index=df_final_clean.index)
    X_final       = pd.concat([X_scaled_part, df_final_clean[['hour_sin','hour_cos']]], axis=1)

    return df_raw, df_before_clean, df_after_clean, df_final_clean, X_final, final_feature_columns


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Upload مرة واحدة بس، الناڤيجيشن لكل الصفحات
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌬️ Air Quality")
    st.markdown("---")

    st.markdown("### 📂 Dataset")
    uploaded_file = st.file_uploader("AirQuality.csv", type=["csv"])

    if uploaded_file:
        file_bytes = uploaded_file.getvalue()           # getvalue() آمن في كل re-run
        if st.session_state['loaded_file_name'] != uploaded_file.name:
            with st.spinner("Running pipeline..."):
                results = load_and_preprocess(file_bytes)
            (st.session_state['df_raw'],
             st.session_state['df_before_clean'],
             st.session_state['df_after_clean'],
             st.session_state['df_final_clean'],
             st.session_state['X_final'],
             st.session_state['feat_cols']) = results
            st.session_state['loaded_file_name'] = uploaded_file.name

        st.success(f"✅ {uploaded_file.name}")
        st.markdown(f"""
        <div style='font-size:0.8rem; color:#8b949e; margin-top:6px;'>
        📊 Raw rows: <b style='color:#58a6ff'>{st.session_state['df_raw'].shape[0]:,}</b><br>
        🧹 After cleaning: <b style='color:#3fb950'>{st.session_state['df_after_clean'].shape[0]:,}</b><br>
        🤖 Ready for clustering: <b style='color:#3fb950'>{st.session_state['df_final_clean'].shape[0]:,}</b>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='color:#8b949e; font-size:0.8rem;'>⬆️ Upload once — data persists across all pages.</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio(
        "",
        ["📂 Preview", "🔍 EDA", "⚙️ Preprocessing", "🤖 Clustering"],
        label_visibility="collapsed"
    )

# ── Helper ────────────────────────────────────────────────────────────────────
data_ready = st.session_state['df_raw'] is not None

def no_data():
    st.markdown(
        '<div class="warn-box">⚠️ Please upload <b>AirQuality.csv</b> from the sidebar first.</div>',
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">🌬️ Air Quality Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Clustering & Exploratory Analysis Dashboard</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📂 Preview":
    if not data_ready:
        no_data()
    else:
        df_raw = st.session_state['df_raw']

        col1, col2, col3, col4 = st.columns(4)
        for col, (val, label) in zip([col1,col2,col3,col4], [
            (df_raw.shape[0],                                        "Total Rows"),
            (df_raw.shape[1],                                        "Total Columns"),
            (int(df_raw.replace(-200, np.nan).isnull().sum().sum()), "Missing Values"),
            (int(df_raw.duplicated().sum()),                         "Duplicate Rows"),
        ]):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{val:,}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">First 10 Rows (Raw)</div>', unsafe_allow_html=True)
        st.dataframe(df_raw.head(10), use_container_width=True)

        st.markdown('<div class="section-header">Statistical Summary</div>', unsafe_allow_html=True)
        st.dataframe(df_raw.replace(-200, np.nan).describe().round(2), use_container_width=True)

        st.markdown(
            '<div class="info-box">💡 Values of <b>-200</b> are missing value markers — replaced with NaN during preprocessing.</div>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 EDA":
    if not data_ready:
        no_data()
    else:
        df_before = st.session_state['df_before_clean']
        df_after  = st.session_state['df_after_clean']

        st.markdown('<div class="info-box">📌 EDA on <b>raw data</b> — after replacing -200 with NaN, before cleaning.</div>', unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["📦 Missing Values", "📈 Pollution Trends", "📅 Temporal Patterns", "🔗 Correlations"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Missing Values Count**")
                missing = df_before.replace(-200, np.nan).isnull().sum().sort_values(ascending=False)
                missing = missing[missing > 0]
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ['#f85149' if v > 1000 else '#d29922' if v > 200 else '#58a6ff' for v in missing.values]
                ax.barh(missing.index, missing.values, color=colors)
                ax.set_xlabel("Missing Count"); ax.set_title("Missing Values per Column")
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col2:
                st.markdown("**Missing Values Pattern**")
                fig, ax = plt.subplots(figsize=(6, 4))
                msno.bar(df_before.replace(-200, np.nan), ax=ax, color='#58a6ff', fontsize=8)
                ax.set_title("Completeness per Column")
                plt.tight_layout(); st.pyplot(fig); plt.close()

            st.markdown("**Missing Values Correlation Heatmap**")
            fig, ax = plt.subplots(figsize=(10, 4))
            msno.heatmap(df_before.replace(-200, np.nan), ax=ax, cmap='Blues')
            ax.set_title("Correlation of Missing Values between Sensors")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with tab2:
            st.markdown("**Outlier Detection — Boxplot**")
            fig, ax = plt.subplots(figsize=(10, 5))
            cols_box = ['PT08.S3(NOx)', 'NOx(GT)', 'PT08.S1(CO)']
            sns.boxplot(data=df_before[cols_box].replace(-200, np.nan), ax=ax,
                        palette=['#58a6ff','#3fb950','#f85149'])
            ax.set_title("Checking for Outliers")
            plt.tight_layout(); st.pyplot(fig); plt.close()

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**CO(GT) Raw Distribution**")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(df_before['CO(GT)'].replace(-200, np.nan).dropna(), bins=50, color='#f85149', ax=ax)
                ax.set_title("CO(GT) Distribution (Raw)"); ax.set_xlabel("CO(GT) Value")
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col2:
                st.markdown("**CO vs NMHC Scatter**")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(df_before['CO(GT)'].replace(-200, np.nan),
                           df_before['NMHC(GT)'].replace(-200, np.nan),
                           alpha=0.3, color='#d29922', s=10)
                ax.set_title("CO vs NMHC (Raw Data)")
                ax.set_xlabel("CO(GT)"); ax.set_ylabel("NMHC(GT)")
                plt.tight_layout(); st.pyplot(fig); plt.close()

        with tab3:
            st.markdown("**Average CO Concentration by Hour of Day**")
            fig, ax = plt.subplots(figsize=(12, 4))
            sns.lineplot(data=df_before, x='Hour', y='CO(GT)', marker='o', color='#58a6ff', ax=ax)
            ax.set_title("Rush Hour Analysis — CO(GT) Throughout the Day")
            ax.set_xticks(range(0, 24)); ax.grid(True, linestyle='--', alpha=0.4)
            ax.set_xlabel("Hour of Day"); ax.set_ylabel("CO(GT) Concentration")
            plt.tight_layout(); st.pyplot(fig); plt.close()

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**NOx by Day of Week**")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(data=df_before, x='Day_of_Week', y='NOx(GT)', ax=ax, palette='coolwarm')
                ax.set_title("Average NOx(GT) by Day"); ax.tick_params(axis='x', rotation=45)
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col2:
                st.markdown("**Benzene (C6H6) by Month**")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.lineplot(data=df_before, x='Month', y='C6H6(GT)', marker='s', color='#f85149', ax=ax)
                ax.set_title("Seasonal Benzene Trend"); ax.set_xticks(range(1, 13))
                plt.tight_layout(); st.pyplot(fig); plt.close()

        with tab4:
            st.markdown("**Correlation Heatmap (After Cleaning)**")
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(df_after.select_dtypes(include=[np.number]).corr(),
                        annot=True, cmap='coolwarm', fmt=".2f",
                        linewidths=0.5, ax=ax, annot_kws={'size': 7})
            ax.set_title("Feature Correlation Heatmap (Post-Cleaning)")
            plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Preprocessing":
    if not data_ready:
        no_data()
    else:
        df_raw    = st.session_state['df_raw']
        df_before = st.session_state['df_before_clean']
        df_after  = st.session_state['df_after_clean']
        df_final  = st.session_state['df_final_clean']

        st.markdown('<div class="section-header">Pipeline Steps</div>', unsafe_allow_html=True)
        steps = [
            ("1️⃣", "Removed last 2 unnamed columns", f"{df_raw.shape[1]} → {df_raw.shape[1]-2} columns"),
            ("2️⃣", "Replaced -200 with NaN",          f"{int((df_raw == -200).sum().sum()):,} values replaced"),
            ("3️⃣", "Removed duplicate rows",          f"{int(df_raw.duplicated().sum())} duplicates dropped"),
            ("4️⃣", "Dropped NMHC(GT), Date, Time",    "3 columns removed"),
            ("5️⃣", "Applied ffill + bfill",            "All NaN values filled"),
            ("6️⃣", "Removed anomalous hours",          "2004-11-03 hours 00:00–07:00 deleted"),
            ("7️⃣", "Clipping outliers",                "CO(GT) @ 5%–95% | PT08.S5(O3) @ 1%–99%"),
        ]
        for icon, step, result in steps:
            col1, col2 = st.columns([2, 1])
            with col1: st.markdown(f"{icon} **{step}**")
            with col2: st.markdown(f"<span style='color:#3fb950; font-size:0.85rem;'>✓ {result}</span>", unsafe_allow_html=True)
            st.markdown("<hr style='border:none; border-top:1px solid #21262d; margin:6px 0'>", unsafe_allow_html=True)

        st.markdown('<div class="section-header">Missing Values — Before vs After</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Before Cleaning**")
            fig, ax = plt.subplots(figsize=(6, 4))
            msno.bar(df_before, ax=ax, color='#f85149', fontsize=7)
            ax.set_title("Before")
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with col2:
            st.markdown("**After Cleaning**")
            fig, ax = plt.subplots(figsize=(6, 4))
            msno.bar(df_after, ax=ax, color='#3fb950', fontsize=7)
            ax.set_title("After")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown('<div class="section-header">Distributions After Preprocessing</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        for (col_name, ax), clr in zip(
            [('CO(GT)',axes[0,0]),('C6H6(GT)',axes[0,1]),('T',axes[1,0]),('RH',axes[1,1])],
            ['#58a6ff','#3fb950','#d29922','#f85149']
        ):
            sns.histplot(df_after[col_name].dropna(), kde=True, ax=ax, color=clr, bins=40)
            ax.set_title(f"{col_name} Distribution")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown('<div class="section-header">Dataset After Preprocessing</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        for col, (val, label) in zip([col1,col2,col3,col4], [
            (df_after.shape[0],                   "Rows After Cleaning"),
            (df_after.shape[1],                   "Columns"),
            (int(df_after.isnull().sum().sum()),  "Remaining NaN"),
            (df_final.shape[0],                   "Ready for Clustering"),
        ]):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{val:,}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Clustering":
    if not data_ready:
        no_data()
    else:
        df_final = st.session_state['df_final_clean']
        X_final  = st.session_state['X_final']

        st.markdown(
            f'<div class="info-box">✅ Using <b>cleaned & feature-engineered data</b> — {len(df_final):,} rows, {X_final.shape[1]} features.</div>',
            unsafe_allow_html=True
        )

        tab1, tab2 = st.tabs(["🔵 KMeans Clustering", "🔴 DBSCAN Clustering"])
        scatter_colors = ['#58a6ff','#3fb950','#f85149','#d29922','#bc8cff','#ff7b72','#ffa657','#79c0ff']

        # ── KMeans ────────────────────────────────────────────────────────────
        with tab1:
            st.markdown("**Parameters**")
            n_clusters = st.slider("Number of Clusters (K)", min_value=2, max_value=8, value=3, step=1)

            with st.spinner("Running KMeans..."):
                inertia = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_final).inertia_
                           for k in range(1, 11)]
                kmeans_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters     = kmeans_final.fit_predict(X_final)
                df_final_km  = df_final.copy()
                df_final_km['Cluster'] = clusters
                sil_score    = silhouette_score(X_final, clusters)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Elbow Method**")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(range(1, 11), inertia, marker='o', linestyle='--', color='#58a6ff', linewidth=2)
                ax.axvline(x=n_clusters, color='#f85149', linestyle=':', linewidth=1.5, label=f'Selected K={n_clusters}')
                ax.set_xlabel("Number of Clusters (K)"); ax.set_ylabel("Inertia")
                ax.set_title("Elbow Method"); ax.legend(); ax.grid(True)
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col2:
                st.markdown("**Cluster Distribution**")
                fig, ax = plt.subplots(figsize=(6, 4))
                counts = df_final_km['Cluster'].value_counts().sort_index()
                ax.bar(counts.index.astype(str), counts.values,
                       color=scatter_colors[:len(counts)], edgecolor='#30363d')
                ax.set_xlabel("Cluster"); ax.set_ylabel("Number of Points")
                ax.set_title("Points per Cluster")
                plt.tight_layout(); st.pyplot(fig); plt.close()

            col1, col2, col3 = st.columns(3)
            for col, (val, label) in zip([col1,col2,col3], [
                (f"{sil_score:.3f}", "Silhouette Score"),
                (n_clusters,         "Clusters"),
                (f"{len(df_final_km):,}", "Rows Clustered"),
            ]):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{val}</div>
                        <div class="metric-label">{label}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("**Cluster Visualization (PCA Space)**")
            fig, ax = plt.subplots(figsize=(10, 5))
            for i in range(n_clusters):
                mask = df_final_km['Cluster'] == i
                ax.scatter(df_final_km.loc[mask,'Sensor_PCA1'], df_final_km.loc[mask,'Sensor_PCA2'],
                           s=15, alpha=0.5, color=scatter_colors[i % len(scatter_colors)], label=f"Cluster {i}")
            ax.set_xlabel("Sensor PCA1"); ax.set_ylabel("Sensor PCA2")
            ax.set_title(f"KMeans Clustering — K={n_clusters}"); ax.legend(markerscale=2)
            plt.tight_layout(); st.pyplot(fig); plt.close()

            st.markdown("**Cluster Analysis Table**")
            analysis = df_final_km.groupby('Cluster').agg({
                'CO(GT)':'mean','NOx(GT)':'mean','T':'mean','Hour':'mean','Is_RushHour':'mean',
            }).round(3).sort_values('CO(GT)')
            analysis.columns = ['Avg CO(GT)','Avg NOx(GT)','Avg Temp','Avg Hour','Rush Hour %']
            analysis['Rush Hour %'] = (analysis['Rush Hour %'] * 100).round(1).astype(str) + '%'
            st.dataframe(analysis, use_container_width=True)

        # ── DBSCAN ────────────────────────────────────────────────────────────
        with tab2:
            st.markdown("**Parameters**")
            col1, col2 = st.columns(2)
            with col1: eps_val  = st.slider("EPS (neighborhood radius)", 1.0, 3.0, 1.8, 0.1)
            with col2: min_samp = st.slider("Min Samples", 5, 50, 22, 1)

            with st.spinner("Running DBSCAN..."):
                nn = NearestNeighbors(n_neighbors=min_samp)
                nn.fit(X_final)
                distances, _ = nn.kneighbors(X_final)
                k_distances  = np.sort(distances[:, -1])

                eps_range   = np.arange(1.0, 2.6, 0.1)
                scores_list = []
                for eps in eps_range:
                    lbs  = DBSCAN(eps=eps, min_samples=min_samp).fit_predict(X_final)
                    mask = lbs != -1
                    nc   = len(set(lbs[mask]))
                    scores_list.append(silhouette_score(X_final[mask], lbs[mask]) if nc > 1 else -1)
                optimal_eps = eps_range[np.argmax(scores_list)]

                db_labels   = DBSCAN(eps=eps_val, min_samples=min_samp).fit_predict(X_final)
                df_final_db = df_final.copy()
                df_final_db['DBSCAN_Cluster'] = db_labels
                n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
                noise_pct     = np.sum(db_labels == -1) / len(db_labels) * 100

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**K-Distance Graph**")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(k_distances, color='#58a6ff', linewidth=1.5)
                ax.axhline(y=1.8, color='#f85149', linestyle='--', label='Suggested Start EPS (1.8)')
                ax.axhline(y=2.2, color='#3fb950', linestyle='--', label='Suggested End EPS (2.2)')
                ax.axhline(y=eps_val, color='#d29922', linestyle='-', linewidth=2, label=f'Selected EPS ({eps_val})')
                ax.set_title(f"K-Distance Graph (min_samples={min_samp})")
                ax.set_xlabel("Points sorted by distance"); ax.set_ylabel(f"{min_samp}-NN Distance")
                ax.legend(fontsize=8)
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col2:
                st.markdown("**DBSCAN Scatter Plot**")
                fig, ax = plt.subplots(figsize=(6, 4))
                dbscan_colors = ['#f85149','#58a6ff','#3fb950','#d29922','#bc8cff','#ff7b72']
                for i, lbl in enumerate(sorted(set(db_labels))):
                    mask = db_labels == lbl
                    ax.scatter(
                        df_final_db.loc[mask,'Sensor_PCA1'], df_final_db.loc[mask,'Sensor_PCA2'],
                        s=10 if lbl==-1 else 15, alpha=0.4 if lbl==-1 else 0.6,
                        color='#21262d' if lbl==-1 else dbscan_colors[i % len(dbscan_colors)],
                        marker='x' if lbl==-1 else 'o',
                        label="Noise (-1)" if lbl==-1 else f"Cluster {lbl}"
                    )
                ax.set_title(f"DBSCAN (eps={eps_val}, min_samples={min_samp})")
                ax.set_xlabel("Sensor PCA1"); ax.set_ylabel("Sensor PCA2")
                ax.legend(markerscale=2, fontsize=8)
                plt.tight_layout(); st.pyplot(fig); plt.close()

            col1, col2, col3 = st.columns(3)
            for col, (val, label) in zip([col1,col2,col3], [
                (n_db_clusters,        "Clusters Found"),
                (f"{noise_pct:.1f}%",  "Noise Points"),
                (f"{optimal_eps:.1f}", "Auto Optimal EPS"),
            ]):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{val}</div>
                        <div class="metric-label">{label}</div>
                    </div>""", unsafe_allow_html=True)

            if n_db_clusters > 0:
                st.markdown("**Cluster Averages (excl. Noise)**")
                db_analysis = df_final_db[df_final_db['DBSCAN_Cluster'] != -1].groupby('DBSCAN_Cluster')[
                    ['CO(GT)','NOx(GT)','T','Is_RushHour']
                ].mean().round(3)
                db_analysis.columns = ['Avg CO(GT)','Avg NOx(GT)','Avg Temp','Rush Hour %']
                db_analysis['Rush Hour %'] = (db_analysis['Rush Hour %'] * 100).round(1).astype(str) + '%'
                st.dataframe(db_analysis, use_container_width=True)