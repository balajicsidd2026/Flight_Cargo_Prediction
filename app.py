import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Flight Cargo Prediction",
    page_icon="✈️",
    layout="wide"
)

# ------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------
st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

/* Header */
.header-box {
    background: linear-gradient(90deg, #dfe9f3, #ffffff);
    padding: 25px;
    border-radius: 20px;
    margin-bottom: 25px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
}

.title-text {
    font-size: 42px;
    font-weight: 700;
    color: #0f172a;
}

.subtitle-text {
    font-size: 20px;
    color: #475569;
}

/* Cards */
.card {
    background: white;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.section-title {
    font-size: 26px;
    font-weight: 600;
    margin-bottom: 15px;
    color: #1e293b;
}

/* Result cards */
.result-card-purple {
    background: linear-gradient(135deg,#ede9fe,#ffffff);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
}

.result-card-blue {
    background: linear-gradient(135deg,#dbeafe,#ffffff);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
}

.result-card-green {
    background: linear-gradient(135deg,#dcfce7,#ffffff);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
}

.metric-title {
    font-size: 24px;
    font-weight: 600;
}

.metric-value {
    font-size: 42px;
    font-weight: 700;
}

/* Button */
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg,#4f46e5,#7c3aed);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px;
    font-size: 22px;
    font-weight: 600;
}

.stButton>button:hover {
    background: linear-gradient(90deg,#4338ca,#6d28d9);
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# LOAD MODELS
# ------------------------------------------------
all_models = joblib.load("model.pkl")
best_model_name = joblib.load("best_model.pkl")

sample_model = all_models[best_model_name]['ULD_Count']

try:
    FEATURES = list(sample_model.feature_names_)
except:
    FEATURES = list(sample_model.feature_names_in_)

# ------------------------------------------------
# MAPPINGS
# ------------------------------------------------
origin_map = {
    "MAA": 0, "AMS": 1, "DXB": 2, "OSL": 3,
    "FRA": 4, "LHR": 5, "CDG": 6, "JFK": 7
}

shc_map = {
    "COL": 0, "RFL": 1, "ELM": 2, "EAW": 3,
    "AOG": 4, "HEA": 5, "SPX": 6, "ECC": 7
}

# ------------------------------------------------
# HELPERS
# ------------------------------------------------
def to_float(x):
    try:
        return float(x)
    except:
        return 0.0


def add_origin_weather_flag(df):

    df['origin_high_wind_flag'] = (df['origin_wind_speed'] > 30).astype(int)
    df['origin_heavy_rain_flag'] = (df['origin_rain'] > 15).astype(int)
    df['origin_snow_flag'] = (df['origin_snowfall'] > 0).astype(int)

    df['origin_bad_weather_flag'] = (
        df['origin_high_wind_flag'] +
        df['origin_heavy_rain_flag'] +
        df['origin_snow_flag']
    ).ge(1).astype(int)

    return df


def preprocess_input(df):

    df['date'] = pd.to_datetime(df['date'])

    df['DayOfWeek'] = df['date'].dt.dayofweek
    df['Month'] = df['date'].dt.month
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

    df = df.rename(columns={
        "origin_Windspeed": "origin_wind_speed",
        "dest_Windspeed": "dest_wind_speed"
    })

    df['origin'] = df['origin'].map(origin_map).fillna(0)
    df['SHC'] = df['SHC'].map(shc_map).fillna(0)

    df = add_origin_weather_flag(df)

    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0

    df = df[FEATURES]

    return df.fillna(0)


def predict_from_input(df_input):

    processed = preprocess_input(df_input)

    model_set = all_models[best_model_name]

    uld = model_set['ULD_Count'].predict(processed)[0]
    pieces = model_set['Total_Pieces'].predict(processed)[0]
    weight = model_set['Total_Weight'].predict(processed)[0]

    return uld, pieces, weight


# ------------------------------------------------
# HEADER
# ------------------------------------------------
st.markdown("""
<div class="header-box">
    <div class="title-text">✈️ Flight Cargo Prediction App</div>
    <div class="subtitle-text">
        Predict ULD Count, Total Pieces & Total Weight
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------
# FLIGHT DETAILS
# ------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Flight Details</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    date = st.date_input("Flight Date")

with col2:
    origin = st.selectbox("Origin", list(origin_map.keys()))

with col3:
    shc = st.selectbox("SHC", list(shc_map.keys()))

with col4:
    holiday_flag = st.selectbox(
        "Holiday Flag",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------
# WEATHER SECTIONS
# ------------------------------------------------
left, right = st.columns(2)

# ORIGIN
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Origin Weather Details</div>',
        unsafe_allow_html=True
    )

    origin_temp = st.text_input("Origin Temperature",value="30")
    origin_rain = st.text_input("Origin Rainfall",value="24")
    origin_wind = st.text_input("Origin Wind Speed",value="13")
    origin_snow = st.text_input("Origin Snowfall",value="2")

    st.markdown('</div>', unsafe_allow_html=True)

# DESTINATION
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Destination Weather Details</div>',
        unsafe_allow_html=True
    )

    dest_temp = st.text_input("Destination Temperature",value="35")
    dest_rain = st.text_input("Destination Rainfall",value="10")
    dest_wind = st.text_input("Destination Wind Speed",value="23")
    dest_snow = st.text_input("Destination Snowfall",value="0")

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------
# INPUT DATA
# ------------------------------------------------
input_df = pd.DataFrame({
    "date": [date],
    "origin": [origin],
    "SHC": [shc],
    "origin_temp": [to_float(origin_temp)],
    "origin_rain": [to_float(origin_rain)],
    "origin_Windspeed": [to_float(origin_wind)],
    "origin_snowfall": [to_float(origin_snow)],
    "dest_temp": [to_float(dest_temp)],
    "dest_rain": [to_float(dest_rain)],
    "dest_Windspeed": [to_float(dest_wind)],
    "dest_snowfall": [to_float(dest_snow)],
    "holiday_flag": [holiday_flag]
})

# ------------------------------------------------
# BUTTON
# ------------------------------------------------
predict_btn = st.button("🚀 Predict Cargo")

# ------------------------------------------------
# RESULTS
# ------------------------------------------------
if predict_btn:

    try:
        uld, pieces, weight = predict_from_input(input_df)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            '<div class="section-title">📊 Prediction Results</div>',
            unsafe_allow_html=True
        )

        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown(f"""
            <div class="result-card-purple">
                <div class="metric-title">ULD Count</div>
                <div class="metric-value">{int(round(uld))}</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class="result-card-blue">
                <div class="metric-title">Total Pieces</div>
                <div class="metric-value">{int(round(pieces))}</div>
            </div>
            """, unsafe_allow_html=True)

        with r3:
            st.markdown(f"""
            <div class="result-card-green">
                <div class="metric-title">Total Weight (kg)</div>
                <div class="metric-value">{round(weight,2)}</div>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
