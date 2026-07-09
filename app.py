import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

import sklearn.neighbors
from sklearn.neighbors import BallTree

from utils import load_data

st.set_page_config(
    page_title="Bhukampa Dristi",
    page_icon="🌍",
    layout="wide"
)

# -------------------------------
# CUSTOM CSS
# -------------------------------

st.markdown("""
<style>

.main{
    background:#f5f7fb;
}

.block-container{
    padding-top:3.5rem;
    padding-bottom:1rem;
}

.title{
    font-size:42px;
    font-weight:800;
    color:#0b5394;
    line-height:1.15;
    margin-top:1.2rem;
    margin-bottom:0.35rem;
    white-space:normal;
    overflow:visible;
}

.subtitle{
    font-size:18px;
    color:#555;
    margin-top:0.25rem;
    line-height:1.3;
    padding:20px;
    border-radius:15px;
    box-shadow:0 3px 10px rgba(0,0,0,.12);
    text-align:center;
}

.prediction-card{
    background:white;
    border-radius:18px;
    padding:25px;
    box-shadow:0 4px 12px rgba(0,0,0,.15);
}

.section{
    background:white;
    border-radius:18px;
    padding:25px;
    margin-top:20px;
    box-shadow:0 3px 12px rgba(0,0,0,.12);
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD DATA
# -------------------------------

df = load_data()

# -------------------------------
# MODEL
# -------------------------------

with open("risk_zone_model.pkl","rb") as f:
    model_bundle = pickle.load(f)

if isinstance(model_bundle, dict):
    model = model_bundle.get("model")
    feature_names = model_bundle.get("features")
    encoder = model_bundle.get("encoder")
else:
    model = model_bundle
    feature_names = None
    encoder = None

if model is None:
    raise ValueError("The saved model bundle does not contain a trained estimator.")

# -------------------------------
# BALL TREE
# -------------------------------

coords = np.radians(
    df[["Latitude","Longitude"]]
)

tree = BallTree(coords, metric="haversine")

EARTH_RADIUS = 6371


def map_prediction_to_label(magnitude, depth, predicted_label):
    label = str(predicted_label).strip().lower()

    if magnitude < 3.0 and depth > 120:
        return "Low"

    if magnitude < 4.0 and depth > 80:
        return "Low"

    if magnitude >= 6.0 and depth <= 70:
        return "High"

    if magnitude >= 4.5 and depth <= 100:
        return "Moderate"

    if label in {"high", "low", "moderate"}:
        return predicted_label

    return "Moderate"

# -------------------------------
# HEADER
# -------------------------------

st.markdown(
"""
<div class='title'>
🌍 Bhukampa Dristi
</div>

<div class='subtitle'>
AI Earthquake Risk Assessment System for Nepal
</div>
""",
unsafe_allow_html=True
)

st.divider()
# ======================================================
# RISK PREDICTOR
# ======================================================

st.markdown("## ⚠️ AI Earthquake Risk Predictor")

left, right = st.columns([1, 1.4])

places = sorted(df["Place"].dropna().unique())

with left:

    st.markdown("### Prediction Input")

    place = st.selectbox(
        "Choose Place",
        places,
        index=0
    )

    magnitude = st.slider(
        "Magnitude",
        min_value=1.0,
        max_value=9.0,
        value=5.5,
        step=0.1
    )

    depth = st.slider(
        "Depth (km)",
        min_value=0,
        max_value=300,
        value=25
    )

    row = df[df["Place"] == place].iloc[0]

    latitude = float(row["Latitude"])
    longitude = float(row["Longitude"])

    kathmandu = (27.7172, 85.3240)
    gorkha = (28.0000, 84.6330)


    def haversine(lat1, lon1, lat2, lon2):

        R = 6371

        dlat = np.radians(lat2-lat1)
        dlon = np.radians(lon2-lon1)

        a = (
            np.sin(dlat/2)**2
            + np.cos(np.radians(lat1))
            * np.cos(np.radians(lat2))
            * np.sin(dlon/2)**2
        )

        return 2 * R * np.arcsin(np.sqrt(a))


    dist_kathmandu = haversine(
        latitude,
        longitude,
        kathmandu[0],
        kathmandu[1]
    )

    dist_gorkha = haversine(
        latitude,
        longitude,
        gorkha[0],
        gorkha[1]
    )


    radius = 50 / EARTH_RADIUS

    count = tree.query_radius(
        np.radians([[latitude, longitude]]),
        r=radius
    )[0]

    nearby = len(count)

    depth_bin = pd.cut(
        [depth],
        bins=[0,30,70,300],
        labels=[0,1,2]
    )[0]

    energy = 10**(1.5*magnitude)

    feature_values = {
        "Latitude": [latitude],
        "Longitude": [longitude],
        "Depth_km": [depth],
        "dist_kathmandu_km": [dist_kathmandu],
        "dist_gorkha_km": [dist_gorkha],
        "quakes_within_50km": [nearby],
        "depth_bin": [int(depth_bin)],
        "energy_proxy": [energy],
    }

    features = pd.DataFrame(feature_values)

    if feature_names is not None:
        features = features.reindex(columns=feature_names, fill_value=0)

    model_input = features.to_numpy(dtype=float)

    if st.button(
        "🔍 Predict Risk",
        use_container_width=True
    ):

        prediction = int(model.predict(model_input)[0])

        if encoder is not None:
            try:
                predicted_label = encoder.inverse_transform([prediction])[0]
            except Exception:
                predicted_label = str(prediction)
        else:
            predicted_label = {0: "High", 1: "Low", 2: "Moderate"}.get(prediction, "Unknown")

        predicted_label = map_prediction_to_label(magnitude, depth, predicted_label)

        if str(predicted_label).lower() == "high":
            risk = "🔴 HIGH"
            color = "#dc2626"
            recommendation = """
            • Immediate preparedness

            • Avoid damaged buildings

            • Keep emergency kit ready

            • Follow official alerts
            """
        elif str(predicted_label).lower() == "moderate":
            risk = "🟡 MODERATE"
            color = "#d97706"
            recommendation = """
            • Stay alert

            • Check emergency exits

            • Monitor updates
            """
        else:
            risk = "🟢 LOW"
            color = "#16a34a"
            recommendation = """
            • Normal activity

            • Keep awareness

            • Learn earthquake safety
            """

        st.markdown("---")
        st.markdown(
            f"<div style='color:{color}; font-size:24px; font-weight:700; margin-bottom:8px;'>{risk}</div>",
            unsafe_allow_html=True,
        )
        st.success(recommendation)

    k1, k2, k3, k4 = st.columns([1.4, 1.6, 1.3, 1.3])

    cards = [
        ("🌍 Total Quakes", len(df), "#2563eb"),
        ("📈 Avg Magnitude", f"{round(df['Magnitude'].mean(), 2):.2f}", "#16a34a"),
        ("⚠ Max Magnitude", round(df['Magnitude'].max(), 2), "#dc2626"),
        ("🌊 Avg Depth", f"{round(df['Depth_km'].mean(), 1)} km", "#ea580c"),
    ]

    for col, (title, value, color) in zip([k1, k2, k3, k4], cards):
        with col:
            st.markdown(
                f"""
                <div style="
                    background:white;
                    border-left:8px solid {color};
                    padding:18px;
                    border-radius:15px;
                    box-shadow:0 8px 20px rgba(0,0,0,.12);
                    color:#111827;
                    min-height:160px;
                    display:flex;
                    flex-direction:column;
                    justify-content:space-between;
                ">
                <div>
                    <h5 style="margin:0 0 12px 0; color:#111827; font-size:16px; line-height:1.1; white-space:normal;">{title}</h5>
                </div>
                <div>
                    <h2 style="margin:0; color:#111827; font-size:36px; white-space:nowrap;">{value}</h2>
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with right:

    st.markdown("### 🗺 Interactive Nepal Map")

    m = folium.Map(
        location=[latitude, longitude],
        zoom_start=8,
        tiles="CartoDB positron"
    )

    color = "green"

    if magnitude >= 7:

        color = "red"

    elif magnitude >= 6:

        color = "orange"

    folium.Circle(

        location=[latitude, longitude],

        radius=magnitude*5000,

        color=color,

        fill=True,

        fill_opacity=0.4

    ).add_to(m)

    folium.Marker(

        [latitude, longitude],

        popup=f"""
        <b>{place}</b><br>
        Magnitude : {magnitude}<br>
        Depth : {depth} km
        """

    ).add_to(m)

    st_folium(
        m,
        width=None,
        height=540
    )

st.divider()
# ======================================================
# HISTORICAL EARTHQUAKE DASHBOARD
# ======================================================

st.markdown("## 📊 Historical Earthquake Dashboard")

# -----------------------------
# KPI CARDS
# -----------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "🌍 Total Earthquakes",
    f"{len(df):,}"
)

c2.metric(
    "📈 Average Magnitude",
    round(df["Magnitude"].mean(), 2)
)

c3.metric(
    "⚡ Maximum Magnitude",
    round(df["Magnitude"].max(), 2)
)

c4.metric(
    "🌊 Average Depth",
    f"{round(df['Depth_km'].mean(),1)} km"
)

st.markdown("")

# -----------------------------
# CHARTS
# -----------------------------

chart1, chart2 = st.columns(2)

with chart1:

    fig = px.histogram(
        df,
        x="Magnitude",
        nbins=25,
        title="Magnitude Distribution",
        color_discrete_sequence=["#2E86DE"]
    )

    fig.update_layout(
        template="plotly_white",
        height=420
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with chart2:

    risk = (
        df["Risk_Level"]
        .value_counts()
        .reset_index()
    )

    risk.columns = ["Risk", "Count"]

    fig = px.pie(
        risk,
        names="Risk",
        values="Count",
        hole=.55,
        title="Risk Distribution"
    )

    fig.update_layout(
        template="plotly_white",
        height=420
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# -----------------------------
# MAGNITUDE VS DEPTH
# -----------------------------

fig = px.scatter(

    df,

    x="Depth_km",

    y="Magnitude",

    color="Risk_Level",

    hover_name="Place",

    size="Magnitude",

    title="Magnitude vs Depth"

)

fig.update_layout(
    template="plotly_white",
    height=520
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------
# YEARLY TREND
# -----------------------------

if "Year" in df.columns:

    yearly = (

        df.groupby("Year")

        .size()

        .reset_index(name="Earthquakes")

    )

    fig = px.line(

        yearly,

        x="Year",

        y="Earthquakes",

        markers=True,

        title="Earthquake Trend"

    )

    fig.update_layout(

        template="plotly_white",

        height=450

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

# -----------------------------
# HEATMAP
# -----------------------------

corr = df[
    [
        "Magnitude",
        "Depth_km",
        "Latitude",
        "Longitude",
        "Risk_Score"
    ]
].corr()

fig = px.imshow(

    corr,

    text_auto=True,

    aspect="auto",

    title="Correlation Heatmap"

)

fig.update_layout(
    height=500
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------
# RECENT RECORDS
# -----------------------------

st.subheader("Latest Earthquake Records")

st.dataframe(

    df[
        [
            "Date",
            "Place",
            "Magnitude",
            "Depth_km",
            "Risk_Level"
        ]
    ].head(20),

    use_container_width=True,

    hide_index=True

)

st.divider()
# ======================================================
# HISTORICAL EARTHQUAKE DASHBOARD
# ======================================================

st.markdown("## 📊 Historical Earthquake Dashboard")

# -----------------------------
# KPI CARDS
# -----------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "🌍 Total Earthquakes",
    f"{len(df):,}"
)

c2.metric(
    "📈 Average Magnitude",
    round(df["Magnitude"].mean(), 2)
)

c3.metric(
    "⚡ Maximum Magnitude",
    round(df["Magnitude"].max(), 2)
)

c4.metric(
    "🌊 Average Depth",
    f"{round(df['Depth_km'].mean(),1)} km"
)

st.markdown("")

# -----------------------------
# CHARTS
# -----------------------------

chart1, chart2 = st.columns(2)

with chart1:

    fig = px.histogram(
        df,
        x="Magnitude",
        nbins=25,
        title="Magnitude Distribution",
        color_discrete_sequence=["#2E86DE"]
    )

    fig.update_layout(
        template="plotly_white",
        height=420
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with chart2:

    risk = (
        df["Risk_Level"]
        .value_counts()
        .reset_index()
    )

    risk.columns = ["Risk", "Count"]

    fig = px.pie(
        risk,
        names="Risk",
        values="Count",
        hole=.55,
        title="Risk Distribution"
    )

    fig.update_layout(
        template="plotly_white",
        height=420
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# -----------------------------
# MAGNITUDE VS DEPTH
# -----------------------------

fig = px.scatter(

    df,

    x="Depth_km",

    y="Magnitude",

    color="Risk_Level",

    hover_name="Place",

    size="Magnitude",

    title="Magnitude vs Depth"

)

fig.update_layout(
    template="plotly_white",
    height=520
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------
# YEARLY TREND
# -----------------------------

if "Year" in df.columns:

    yearly = (

        df.groupby("Year")

        .size()

        .reset_index(name="Earthquakes")

    )

    fig = px.line(

        yearly,

        x="Year",

        y="Earthquakes",

        markers=True,

        title="Earthquake Trend"

    )

    fig.update_layout(

        template="plotly_white",

        height=450

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

# -----------------------------
# HEATMAP
# -----------------------------

corr = df[
    [
        "Magnitude",
        "Depth_km",
        "Latitude",
        "Longitude",
        "Risk_Score"
    ]
].corr()

fig = px.imshow(

    corr,

    text_auto=True,

    aspect="auto",

    title="Correlation Heatmap"

)

fig.update_layout(
    height=500
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------
# RECENT RECORDS
# -----------------------------

st.subheader("Latest Earthquake Records")

st.dataframe(

    df[
        [
            "Date",
            "Place",
            "Magnitude",
            "Depth_km",
            "Risk_Level"
        ]
    ].head(20),

    use_container_width=True,

    hide_index=True

)

st.divider()
st.divider()

st.markdown(
"""
<div style='text-align:center;padding:20px;'>

<h2>🌍 Bhukampa Dristi</h2>

AI Earthquake Risk Assessment System for Nepal

Built using

<b>Python • Streamlit • Machine Learning • Random Forest • Plotly • Folium</b>

</div>
""",
unsafe_allow_html=True
)
fig = go.Figure(go.Indicator(

mode="gauge+number",

value=magnitude,

title={"text":"Earthquake Severity"},

gauge={

'axis':{'range':[1,9]},

'bar':{'color':'darkred'},

'steps':[

{'range':[1,4],'color':'green'},

{'range':[4,6],'color':'yellow'},

{'range':[6,9],'color':'red'}

]

}

))

st.plotly_chart(fig,use_container_width=True)
proba = model.predict_proba(model_input)[0]

prob=pd.DataFrame({

"Risk":["Low","Medium","High"],

"Probability":proba

})

fig=px.bar(

prob,

x="Risk",

y="Probability",

color="Risk"

)

st.plotly_chart(fig,use_container_width=True)
st.subheader("Latest Earthquake Timeline")

latest=df.head(10)

fig=px.scatter(

latest,

x="Date",

y="Magnitude",

size="Magnitude",

color="Risk_Level",

hover_name="Place"

)

st.plotly_chart(fig,use_container_width=True)
if prediction==2:

    st.error("🔴 HIGH RISK")

elif prediction==1:

    st.warning("🟡 MEDIUM RISK")

else:

    st.success("🟢 LOW RISK")
st.markdown("---")

st.markdown("""

<center>

### 🌍 Bhukampa Dristi

AI Earthquake Risk Assessment System

Machine Learning • Expert System • Streamlit • Plotly • Folium

Developed by **Abinash Oli**

</center>

""",unsafe_allow_html=True)