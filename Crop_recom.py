# crop_recommendation_system.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import io
import joblib
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(page_title="Crop Recommendation System", layout="wide", page_icon="🌾")

# ---------------------------------------------------------------
# CUSTOM CSS STYLE
# ---------------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    .main {
        background: linear-gradient(135deg, #e3f2fd 0%, #f9fff5 100%);
        padding: 1rem 2rem;
    }
    .left-panel {
        background: #4caf50;
        color: white;
        padding: 2rem;
        border-radius: 16px;
        height: 95vh;
    }
    .left-panel h1 {
        font-size: 2.4rem;
        font-weight: 700;
        color: #e8f5e9;
    }
    .left-panel p {
        font-size: 0.95rem;
        line-height: 1.6;
        color: #f1f8e9;
    }
    .predict-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
    }
    .chart-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    h2 {
        color: #1b5e20;
        font-weight: 600;
    }
    .footer-note {
        font-size: 13px;
        color: #616161;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# LEFT PANEL CONTENT
# ---------------------------------------------------------------
left_col, right_col = st.columns([0.4, 0.6])

with left_col:
    st.markdown("<div class='left-panel'>", unsafe_allow_html=True)
    st.markdown("<h1>🌾 Crop Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("""
        <h4>Using Random Forest Algorithm</h4>
        <p>
        This intelligent crop recommendation system analyzes soil nutrients (N, P, K), temperature,
        humidity, pH, and rainfall to recommend the best crop for cultivation using 
        <b>Machine Learning</b>. 
        </p>
        <p>
        The model is trained using the Random Forest algorithm on an Indian crop dataset. 
        By providing environmental inputs, the system suggests the most suitable crop 
        for optimal yield.
        </p>
        <p><b>Features used:</b> Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall.</p>
        <br>
        <p><i>Empowering smart farming through data-driven insights 🌱</i></p>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# RIGHT PANEL – PREDICTION FORM & VISUALIZATIONS
# ---------------------------------------------------------------
with right_col:
    st.markdown("<div class='predict-card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;'>Please input the feature values to predict the best crop to plant.</h2>", unsafe_allow_html=True)

    # Sample or uploaded dataset
    uploaded_file = st.file_uploader("Upload dataset (CSV with N,P,K,temperature,humidity,ph,rainfall,label)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset uploaded successfully!")
    else:
        # Generate demo dataset
        crops = ["rice","maize","chickpea","kidneybeans","pigeonpeas","mothbeans","mungbean","blackgram","lentil","pomegranate","banana","mango"]
        rng = np.random.RandomState(42)
        rows = 400
        df = pd.DataFrame({
            "N": rng.randint(0, 200, rows),
            "P": rng.randint(0, 200, rows),
            "K": rng.randint(0, 200, rows),
            "temperature": rng.uniform(10, 40, rows).round(1),
            "humidity": rng.uniform(20, 100, rows).round(1),
            "ph": rng.uniform(3.5, 9.5, rows).round(1),
            "rainfall": rng.uniform(20, 300, rows).round(1),
            "label": rng.choice(crops, rows)
        })
        st.info("Using demo dataset. Upload a CSV for real analysis.")

    # Encode labels
    le = LabelEncoder()
    df['label_enc'] = le.fit_transform(df['label'])
    X = df[["N","P","K","temperature","humidity","ph","rainfall"]]
    y = df['label_enc']

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    # Prediction form
    st.markdown("### Enter feature values")
    c1, c2 = st.columns(2)
    with c1:
        N = st.number_input("Insert N (kg/ha)", min_value=0.0, max_value=200.0, value=50.0)
        P = st.number_input("Insert P (kg/ha)", min_value=0.0, max_value=200.0, value=40.0)
        K = st.number_input("Insert K (kg/ha)", min_value=0.0, max_value=200.0, value=50.0)
        temp = st.number_input("Insert Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
    with c2:
        humidity = st.number_input("Insert Avg Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
        ph_val = st.number_input("Insert pH value", min_value=0.0, max_value=14.0, value=6.5)
        rainfall = st.number_input("Insert Avg Rainfall (mm)", min_value=0.0, max_value=400.0, value=120.0)
        submit = st.button("🔍 Predict Crop")

    if submit:
        user_input = np.array([[N, P, K, temp, humidity, ph_val, rainfall]])
        pred = model.predict(user_input)[0]
        crop_name = le.inverse_transform([pred])[0]
        st.success(f"🌾 **Recommended Crop:** {crop_name}")

        # Probability chart
        probs = model.predict_proba(user_input)[0]
        dfp = pd.DataFrame({"Crop": le.inverse_transform(np.arange(len(probs))), "Probability": probs})
        figp = px.bar(dfp.sort_values("Probability", ascending=False), x="Crop", y="Probability", color="Crop", title="Prediction Probability Distribution")
        st.plotly_chart(figp, use_container_width=True)

    st.markdown(f"**Model Accuracy:** {acc:.3f}")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------------
    # VISUALIZATION SECTION
    # ---------------------------------------------------------------
    st.write("---")
    st.markdown("## 📊 Data Visualizations")

    colA, colB = st.columns(2)

    # Correlation heatmap
    with colA:
        st.markdown("### Correlation Heatmap")
        fig_corr = px.imshow(X.corr(), text_auto=".2f", color_continuous_scale="Greens", title="Feature Correlation")
        st.plotly_chart(fig_corr, use_container_width=True)

    # Feature importance
    with colB:
        st.markdown("### Feature Importance")
        importance = model.named_steps['clf'].feature_importances_
        imp_df = pd.DataFrame({"Feature": X.columns, "Importance": importance}).sort_values("Importance", ascending=False)
        fig_imp = px.bar(imp_df, x="Feature", y="Importance", color="Feature", title="Feature Importance (Random Forest)")
        st.plotly_chart(fig_imp, use_container_width=True)

    # Distribution plots
    st.markdown("### Feature Distributions")
    df_melt = X.melt(var_name="Feature", value_name="Value")
    fig_dist = px.violin(df_melt, x="Feature", y="Value", box=True, points="all", color="Feature", title="Distribution of Input Features")
    st.plotly_chart(fig_dist, use_container_width=True)

    # Confusion matrix
    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig_cm = px.imshow(cm, x=le.classes_, y=le.classes_, text_auto=True, color_continuous_scale="Greens", title="Confusion Matrix (Test Set)")
    st.plotly_chart(fig_cm, use_container_width=True)

    # Classification report
    st.markdown("### Classification Report")
    report = classification_report(y_test, model.predict(X_test), target_names=le.classes_, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)

    # Footer
    st.markdown("<div class='footer-note'>© 2025 Crop Recommendation System | Built using Streamlit, Scikit-learn, and Plotly</div>", unsafe_allow_html=True)