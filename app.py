# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from main import run_system
import tempfile
from matplotlib_venn import venn3
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Enterprise Insider Threat Detection", layout="wide")

# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Select Section", ["📁 Upload & Detect", "📊 Summary & Graphs", "🚨 Anomalies", "📈 Compare Models"])

# Load CSV file
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

uploaded_file = st.sidebar.file_uploader("📄 Upload Behavioral Log CSV", type="csv")
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(uploaded_file.read())
        data_path = tmp_file.name
else:
    data_path = "complex_user_data.csv"

# Upload & Detect Tab
if page == "📁 Upload & Detect":
    st.title("🛡️ Insider Threat Detection System")

    model_type = st.selectbox("🧠 Select Anomaly Detection Model", ["isolation_forest", "one_class_svm", "autoencoder"])

    if st.button("🚀 Run Detection"):
        with st.spinner("Analyzing user behavior..."):
            result_df = run_system(data_path, model_type)
            st.session_state['results'] = result_df
            st.success("✅ Detection Complete")

            anomalies = result_df[result_df['predicted_anomaly'] == 1]
            if not anomalies.empty:
                st.toast(f"⚠️ {len(anomalies)} anomalies detected!", icon="⚠️")
                st.markdown(f"<div style='background-color:#000000;padding:15px;border-radius:10px;font-size:18px;'><b>🚨 {len(anomalies)} anomalies detected!</b> Review the anomalies tab for details.</div>", unsafe_allow_html=True)
            else:
                st.toast("✅ No anomalies found.", icon="✅")
                st.markdown("<div style='background-color:#e6ffe6;padding:15px;border-radius:10px;font-size:18px;'><b>No anomalies found.</b> Everything looks normal.</div>", unsafe_allow_html=True)

# Summary & Graphs Tab
elif page == "📊 Summary & Graphs":
    st.title("📊 Behavioral Analytics Dashboard")

    if 'results' not in st.session_state:
        st.warning("⚠️ Please run detection first.")
    else:
        df = st.session_state['results']

        st.subheader("1. User Distribution by Department")
        fig1 = px.histogram(df, x='department', color='predicted_anomaly',
                            barmode='group', title="User Distribution per Department")
        st.plotly_chart(fig1, use_container_width=True)
        st.caption("This graph shows the distribution of users across departments. Anomalous behavior appears in red.")

        st.subheader("2. Login Hour Heatmap")
        heatmap_data = df.pivot_table(index='login_hour', columns='department', aggfunc='size', fill_value=0)
        fig2 = px.imshow(heatmap_data, text_auto=True, aspect="auto", title="Login Hour by Department")
        st.plotly_chart(fig2)
        st.caption("This heatmap reveals peak login hours per department. Odd login times might indicate threats.")

        st.subheader("3. Anomaly Ratio Pie Chart")
        pie_data = df['predicted_anomaly'].value_counts()
        labels = ['Normal', 'Anomaly']
        fig3 = px.pie(names=labels, values=pie_data, title="Anomaly vs Normal Behavior")
        st.plotly_chart(fig3)
        st.caption("Proportion of anomalies vs normal behavior based on model prediction.")

        st.subheader("4. Boxplot of Key Behavioral Metrics")
        fig4 = px.box(df, y=['login_duration', 'file_download_mb', 'failed_logins'],
                      title="Distribution of Key Metrics")
        st.plotly_chart(fig4)
        st.caption("Analyzes variations in login duration, file downloads, and failed login attempts.")

        st.subheader("5. VPN & Device Usage Patterns")
        fig5 = px.sunburst(df, path=['vpn_usage', 'external_device_access', 'predicted_anomaly'],
                           title="VPN and Device Access Behavior")
        st.plotly_chart(fig5)
        st.caption("Shows nested behavioral access patterns that might lead to anomalies.")

# Anomalies Tab
elif page == "🚨 Anomalies":
    st.title("🚨 Detected Anomalies")

    if 'results' not in st.session_state:
        st.warning("⚠️ Please run detection first.")
    else:
        df = st.session_state['results']
        anomalies = df[df['predicted_anomaly'] == 1]

        st.markdown(f"🔍 <b>{len(anomalies)} anomalies detected</b>", unsafe_allow_html=True)
        st.dataframe(anomalies)

        st.subheader("🔍 Pattern in Anomalies - Heatmap of Suspicious Features")
        heat_df = anomalies[['login_hour', 'file_download_mb', 'failed_logins', 'number_of_devices_used']]
        fig6 = px.imshow(heat_df.corr(), text_auto=True, color_continuous_scale='Reds')
        st.plotly_chart(fig6)
        st.caption("Heatmap reveals strong correlations that are common in anomalous cases.")

        st.subheader("📉 File Downloads in Anomalies")
        fig7 = px.histogram(anomalies, x='file_download_mb', nbins=40, title="File Download Volume in Anomalies")
        st.plotly_chart(fig7)
        st.caption("Most anomalies involve large file downloads, especially off-hours or on VPN.")

# Compare Models Tab
elif page == "📈 Compare Models":
    st.title("📈 Model Comparison Overview")

    models = ["isolation_forest", "one_class_svm", "autoencoder"]
    scores = {}
    compare_outputs = {}

    for model in models:
        df_model = run_system(data_path, model)
        true = df_model['is_anomaly']
        pred = df_model['predicted_anomaly']
        scores[model] = {
            "Accuracy": accuracy_score(true, pred),
            "Precision": precision_score(true, pred, zero_division=0),
            "Recall": recall_score(true, pred, zero_division=0),
            "F1 Score": f1_score(true, pred, zero_division=0),
        }
        compare_outputs[model] = pred

    metrics_df = pd.DataFrame(scores).T
    best_model = metrics_df['F1 Score'].idxmax()

    st.subheader("📊 Performance Metrics (Higher is Better)")
    st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen'))
    st.caption("Table shows Accuracy, Precision, Recall, and F1-Score for each model. The best performer is highlighted.")

    st.subheader("🏆 Best Performing Model")
    st.success(f"✅ {best_model} performed best based on F1 Score.")

    st.subheader("📈 Model Anomaly Count")
    count_data = {model: compare_outputs[model].sum() for model in models}
    fig8 = px.bar(x=list(count_data.keys()), y=list(count_data.values()), labels={'x': 'Model', 'y': 'Anomalies'},
                  title="Total Anomalies Detected per Model")
    st.plotly_chart(fig8)
    st.caption("This bar chart shows which model is more/less aggressive in detecting anomalies.")

    st.subheader("📊 Model Agreement (Top 100 Predictions)")
    df_compare = pd.DataFrame(compare_outputs)
    df_compare = df_compare.iloc[:100]
    df_compare['Agreement'] = df_compare.apply(lambda x: len(set(x)) == 1, axis=1)
    agreement_ratio = df_compare['Agreement'].mean()
    st.metric("Agreement on Top 100", f"{agreement_ratio * 100:.2f}%")

    venn_sets = [set(np.where(pred[:100] == 1)[0]) for pred in compare_outputs.values()]
    fig9, ax = plt.subplots()
    venn3(subsets=venn_sets, set_labels=models, ax=ax)
    st.pyplot(fig9)
    st.caption("The Venn diagram visualizes the overlap in anomalies detected by the top 100 samples.")

# Footer
st.markdown("---")
st.markdown("© 2025 Enterprise AI-Based Insider Threat Detection System | Developed By Muhammad Ali.")
