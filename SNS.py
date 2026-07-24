import streamlit as st
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Wine Quality", layout="wide")

@st.cache_data
def load_and_train():
    df = pd.read_csv('juice.csv')
    
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    corr = df.corr()['quality'].abs().sort_values()
    features_under = corr.head(2).index.tolist()
    
    X_train_under = X_train[features_under]
    
    model_under = GaussianNB()
    model_under.fit(X_train_under, y_train)
    
    model_over = GaussianNB(var_smoothing=1e-9)
    model_over.fit(X_train_scaled, y_train)
    
    selector = SelectKBest(score_func=f_classif, k=6)
    X_train_opt = selector.fit_transform(X_train_scaled, y_train)
    
    model_opt = GaussianNB(var_smoothing=0.001)
    model_opt.fit(X_train_opt, y_train)
    
    return df, model_under, model_over, model_opt, scaler, selector, features_under, encoder, X.columns.tolist()

df, model_under, model_over, model_opt, scaler, selector, features_under, encoder, feature_names = load_and_train()

st.title("Wine Quality Classification")
st.write("Naive Bayes: Underfit vs Overfit vs Optimal")

st.sidebar.header("Wine Characteristics")

feature_ranges = {
    'fixed acidity': (4.0, 16.0, 7.5),
    'volatile acidity': (0.0, 1.6, 0.5),
    'citric acid': (0.0, 1.6, 0.3),
    'residual sugar': (0.0, 70.0, 2.5),
    'chlorides': (0.0, 0.7, 0.05),
    'free sulfur dioxide': (0.0, 300.0, 30.0),
    'total sulfur dioxide': (0.0, 500.0, 150.0),
    'density': (0.98, 1.04, 0.995),
    'pH': (2.5, 4.5, 3.2),
    'sulphates': (0.0, 2.0, 0.5),
    'alcohol': (8.0, 15.0, 10.0)
}

feature_vals = {}
for feature, (min_val, max_val, default) in feature_ranges.items():
    if feature == 'density':
        feature_vals[feature] = st.sidebar.slider(
            feature,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default),
            step=0.0005,
            format="%.4f"
        )
    else:
        feature_vals[feature] = st.sidebar.slider(
            feature.capitalize(),
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default),
            step=0.1
        )

predict_btn = st.sidebar.button("Predict", type="primary")

st.sidebar.markdown("---")
st.sidebar.subheader("Models")
st.sidebar.write("Underfit: 2 features only")
st.sidebar.write("Overfit: All features + low smoothing")
st.sidebar.write("Optimal: Feature selection + tuning")

col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("Quality Distribution")
    fig = px.histogram(df, x='quality', title='')
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    corr = df.corr()['quality'].sort_values(ascending=False)
    fig = px.bar(x=corr.values, y=corr.index, title='Correlation with Quality', orientation='h')
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Dataset Statistics")
    st.metric("Samples", f"{df.shape[0]}")
    st.metric("Features", f"{df.shape[1] - 1}")
    st.metric("Quality Classes", f"{df['quality'].nunique()}")
    st.metric("Mean Quality", f"{df['quality'].mean():.2f}")
    st.metric("Quality Range", f"{df['quality'].min()} - {df['quality'].max()}")

def predict(features_array):
    under_features = features_array[:, [feature_names.index(f) for f in features_under]]
    pred_under = model_under.predict(under_features)
    proba_under = model_under.predict_proba(under_features)
    
    scaled = scaler.transform(features_array)
    pred_over = model_over.predict(scaled)
    proba_over = model_over.predict_proba(scaled)
    
    scaled_opt = scaler.transform(features_array)
    opt_features = selector.transform(scaled_opt)
    pred_opt = model_opt.predict(opt_features)
    proba_opt = model_opt.predict_proba(opt_features)
    
    return {
        'under': {'pred': encoder.inverse_transform(pred_under)[0], 'proba': proba_under[0]},
        'over': {'pred': encoder.inverse_transform(pred_over)[0], 'proba': proba_over[0]},
        'opt': {'pred': encoder.inverse_transform(pred_opt)[0], 'proba': proba_opt[0]}
    }

if predict_btn:
    st.markdown("---")
    st.subheader("Results")
    
    features_array = np.array([list(feature_vals.values())])
    results = predict(features_array)
    classes = sorted(df['quality'].unique())
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Underfit")
        pred = results['under']['pred']
        proba = results['under']['proba']
        st.metric("Prediction", pred, f"Conf: {max(proba):.1%}")
        
        proba_full = np.zeros(len(classes))
        for i, cls in enumerate(classes):
            if cls in encoder.classes_:
                idx = np.where(encoder.classes_ == cls)[0][0]
                proba_full[i] = proba[idx]
        
        fig = go.Figure(data=[go.Bar(x=classes, y=proba_full, text=[f"{p:.1%}" for p in proba_full], textposition='auto')])
        fig.update_layout(title="", xaxis_title="Quality", yaxis_title="Probability", yaxis_range=[0, 1], height=250, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Overfit")
        pred = results['over']['pred']
        proba = results['over']['proba']
        st.metric("Prediction", pred, f"Conf: {max(proba):.1%}")
        
        proba_full = np.zeros(len(classes))
        for i, cls in enumerate(classes):
            if cls in encoder.classes_:
                idx = np.where(encoder.classes_ == cls)[0][0]
                proba_full[i] = proba[idx]
        
        fig = go.Figure(data=[go.Bar(x=classes, y=proba_full, text=[f"{p:.1%}" for p in proba_full], textposition='auto')])
        fig.update_layout(title="", xaxis_title="Quality", yaxis_title="Probability", yaxis_range=[0, 1], height=250, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown("### Optimal")
        pred = results['opt']['pred']
        proba = results['opt']['proba']
        st.metric("Prediction", pred, f"Conf: {max(proba):.1%}")
        
        proba_full = np.zeros(len(classes))
        for i, cls in enumerate(classes):
            if cls in encoder.classes_:
                idx = np.where(encoder.classes_ == cls)[0][0]
                proba_full[i] = proba[idx]
        
        fig = go.Figure(data=[go.Bar(x=classes, y=proba_full, text=[f"{p:.1%}" for p in proba_full], textposition='auto')])
        fig.update_layout(title="", xaxis_title="Quality", yaxis_title="Probability", yaxis_range=[0, 1], height=250, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Comparison")
    
    comp_data = {
        'Model': ['Underfit', 'Overfit', 'Optimal'],
        'Prediction': [results['under']['pred'], results['over']['pred'], results['opt']['pred']],
        'Confidence': [max(results['under']['proba']), max(results['over']['proba']), max(results['opt']['proba'])]
    }
    comp_df = pd.DataFrame(comp_data)
    
    fig = go.Figure(data=[go.Bar(x=comp_df['Model'], y=comp_df['Confidence'], text=[f"{c:.1%}" for c in comp_df['Confidence']], textposition='auto')])
    fig.update_layout(title="Confidence Comparison", xaxis_title="Model", yaxis_title="Confidence", yaxis_range=[0, 1], height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Probabilities by Class")
    
    detail_data = {'Quality': sorted(df['quality'].unique())}
    for name in ['under', 'over', 'opt']:
        proba = results[name]['proba']
        proba_full = np.zeros(len(detail_data['Quality']))
        for i, cls in enumerate(detail_data['Quality']):
            if cls in encoder.classes_:
                idx = np.where(encoder.classes_ == cls)[0][0]
                proba_full[i] = proba[idx]
        if name == 'under':
            detail_data['Underfit'] = proba_full
        elif name == 'over':
            detail_data['Overfit'] = proba_full
        else:
            detail_data['Optimal'] = proba_full
    
    detail_df = pd.DataFrame(detail_data)
    detail_df = detail_df.style.format({col: '{:.1%}' for col in detail_df.columns if col != 'Quality'})
    st.dataframe(detail_df, use_container_width=True)
    
    st.markdown("---")
    
    best = max([('Underfit', max(results['under']['proba'])), ('Overfit', max(results['over']['proba'])), ('Optimal', max(results['opt']['proba']))], key=lambda x: x[1])
    
    if best[0] == 'Optimal':
        st.success(f"Recommended: {best[0]} ({best[1]:.1%} confidence)")
    elif best[0] == 'Overfit':
        st.warning(f"{best[0]} has {best[1]:.1%} confidence but may not generalize")
    else:
        st.warning(f"{best[0]} has {best[1]:.1%} confidence - model too simple")

else:
    st.info("Adjust parameters in sidebar and click Predict")

st.markdown("---")
st.markdown("**Models:** Underfit (2 features) | Overfit (all features) | Optimal (selected + tuned)")