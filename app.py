import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.gridspec import GridSpec
from sklearn.ensemble import GradientBoostingClassifier  
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
TARGET_NAMES = {0: "Dropout", 1: "Graduate"}
PASTEL = sns.color_palette("pastel")

st.set_page_config(
    page_title="Student Dropout & Academic Success",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid")

@st.cache_data(show_spinner=False)
def load_raw_data():
    return pd.read_csv(DATA_PATH, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def build_pipeline():
    data = load_raw_data()

    data_filtered = data[data["Target"] != "Enrolled"]
    data_binary = data_filtered.copy()
    data_binary["Target"] = data_binary["Target"].map({"Dropout": 0, "Graduate": 1})
    corr_matrix = data_binary.corr()
    positive_features = corr_matrix["Target"][corr_matrix["Target"] > 0].index
    data_positive = data_binary[positive_features]
    data_clean = data_positive.copy()
    for col in data_clean.columns[:-1]:
        data_clean = remove_outliers_iqr(data_clean, col, 10)
    return {
        "raw": data,
        "binary": data_binary,
        "corr_matrix": corr_matrix,
        "positive": data_positive,
        "clean": data_clean,
        "feature_cols": list(data_clean.columns[:-1]),
    }

def remove_outliers_iqr(df, column, k):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return df[(df[column] >= lower) & (df[column] <= upper)]


@st.cache_resource(show_spinner="Training models…")
def train_models():

    data_clean = build_pipeline()["clean"]

    X = data_clean.drop("Target", axis=1)
    y = data_clean["Target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    feature_names = list(X_train.columns.values)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    results, models = [], {}
    grid = GridSearchCV(
        LogisticRegression(random_state=42),
        {"C": [0.001, 0.01, 0.1, 1, 10, 100]},
        cv=5,
        scoring="accuracy",
    )
    grid.fit(X_train_s, y_train)
    best_c = grid.best_params_["C"]
    logistic = LogisticRegression(C=best_c, random_state=42)
    logistic.fit(X_train_s, y_train)
    models["Logistic Regression"] = logistic
    results.append(score_model("Logistic Regression", y_test, logistic.predict(X_test_s)))

    nb = GaussianNB()
    nb.fit(X_train_s, y_train)
    models["Naive Bayes"] = nb
    results.append(score_model("Naive Bayes", y_test, nb.predict(X_test_s)))
    linreg = LinearRegression()
    linreg.fit(X_train_s, y_train)
    y_pred_linear = (linreg.predict(X_test_s) >= 0.5).astype(int)
    models["Linear Regression"] = linreg
    results.append(score_model("Linear Regression", y_test, y_pred_linear))

    results_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "y_pred"} for r in results]
    ).sort_values(by="Accuracy", ascending=False)
    return {
        "models": models,
        "scaler": scaler,
        "feature_names": feature_names,
        "results": results,
        "results_df": results_df,
        "best_c": best_c,
        "y_test": y_test,
        "X_test_s": X_test_s,
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "train_counts": (int((y_train == 0).sum()), int((y_train == 1).sum())),
    }


def score_model(name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    return {
        "Algorithm": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "y_pred": y_pred,
    }


@st.cache_resource(show_spinner=False)
def load_saved_artifacts():
    required = [
        "logistic_regression_model.pkl",
        "scaler.pkl",
        "feature_names.pkl",
    ]
    if not all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in required):
        return None
    names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    return {
        "model": joblib.load(os.path.join(MODEL_DIR, "logistic_regression_model.pkl")),
        "scaler": joblib.load(os.path.join(MODEL_DIR, "scaler.pkl")),
        "feature_names": [str(n).lstrip("﻿") for n in names],
    }


def get_predictor():
    saved = load_saved_artifacts()
    if saved is not None:
        return saved["model"], saved["scaler"], saved["feature_names"], "saved .pkl files"
    trained = train_models()
    return (
        trained["models"]["Logistic Regression"],
        trained["scaler"],
        trained["feature_names"],
        "model trained in this session",
    )


def show_fig(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def labelled_bars(ax, labels, values, palette=PASTEL):
    bars = ax.bar(list(labels), list(values), color=list(palette)[: len(labels)])
    ax.bar_label(bars, labels=list(values), fontsize=10, color="black")
    return bars

def page_overview(pipe):
    data = pipe["raw"]
    st.title("🎓 Student Dropout & Academic Success Prediction")
    st.caption(
        "Predicting whether a student will drop out or graduate, from academic, "
        "demographic and macro-economic data collected at enrollment."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", f"{data.shape[0]:,}")
    c2.metric("Features", data.shape[1] - 1)
    c3.metric("Missing values", int(data.isnull().sum().sum()))
    c4.metric("Duplicate rows", int(data.duplicated().sum()))

    st.divider()
    st.subheader("Dataset preview")
    st.dataframe(data.head(50), width="stretch", height=320)

    tab1, tab2, tab3 = st.tabs(["Summary statistics", "Column types", "Missing values"])
    with tab1:
        st.dataframe(data.describe().round(3).T, width="stretch", height=420)
    with tab2:
        info = pd.DataFrame(
            {
                "Column": data.columns,
                "Non-null count": data.notnull().sum().values,
                "Dtype": data.dtypes.astype(str).values,
                "Unique values": [data[c].nunique() for c in data.columns],
            }
        )
        st.dataframe(info, width="stretch", height=420, hide_index=True)
    with tab3:
        nulls = data.isnull().sum().rename("Missing").to_frame()
        st.dataframe(nulls, width="stretch", height=420)

    st.divider()
    st.subheader("Preprocessing pipeline")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Raw rows", f"{pipe['raw'].shape[0]:,}")
    s2.metric(
        "After dropping 'Enrolled'",
        f"{pipe['binary'].shape[0]:,}",
        delta=f"{pipe['binary'].shape[0] - pipe['raw'].shape[0]:,}",
    )
    s3.metric(
        "Positively correlated features",
        pipe["positive"].shape[1] - 1,
        delta=f"{pipe['positive'].shape[1] - pipe['raw'].shape[1]:,}",
    )
    s4.metric(
        "After outlier removal",
        f"{pipe['clean'].shape[0]:,}",
        delta=f"{pipe['clean'].shape[0] - pipe['binary'].shape[0]:,}",
    )
    st.markdown(
        "1. Drop the `Enrolled` class — the task becomes **Dropout (0) vs Graduate (1)**.\n"
        "2. Keep only features with a **positive correlation** to the target.\n"
        "3. Remove outliers with the **IQR rule (k = 10)**.\n"
        "4. Standardise features, then split 67 / 33 for train / test."
    )


def page_eda(pipe):
    data = pipe["raw"]
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Target distribution")
    counts = data["Target"].value_counts()
    labels = ["Dropout", "Enrolled", "Graduate"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    ax1.pie(counts, autopct="%.2f%%", labels=labels, colors=PASTEL)
    ax1.set_title("Percentage of students")
    labelled_bars(ax2, labels, counts.values)
    ax2.set(xlabel="Target Categories", ylabel="Number of students")
    ax2.set_title("Total number of students")
    show_fig(fig)

    st.divider()
    st.subheader("Demographics")
    left, right = st.columns(2)

    with left:
        gender_counts = data["Gender"].value_counts()
        gender_labels = gender_counts.index.map({0: "Female", 1: "Male"})
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        ax1.pie(gender_counts, labels=gender_labels, autopct="%.2f%%", colors=PASTEL)
        ax1.set_title("Distribution of Gender")
        labelled_bars(ax2, gender_labels, gender_counts.values)
        ax2.set(xlabel="Gender", ylabel="Number of students")
        ax2.set_title("Distribution of Gender (Bar Plot)")
        show_fig(fig)

    with right:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        sns.countplot(data=data, x="Gender", hue="Target", order=[0, 1],
                      hue_order=["Enrolled", "Dropout", "Graduate"], ax=ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Female", "Male"])
        ax.set_ylabel("Number of students")
        ax.set_title("Outcome by gender")
        show_fig(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    sns.countplot(data=data, x="Marital status", hue="Target", ax=ax)
    ax.set_xticks(range(6))
    ax.set_xticklabels(
        ["Single", "Married", "Widower", "Divorced", "Facto Union", "Legally Separated"]
    )
    ax.set_xlabel("Marital Status")
    ax.set_ylabel("How many Students")
    ax.set_title("Outcome by marital status")
    show_fig(fig)

    st.divider()
    st.subheader("Explore any feature")
    feature = st.selectbox(
        "Feature", [c for c in data.columns if c != "Target"], index=18
    )
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.histplot(data=data, x=feature, hue="Target", kde=True,
                     palette="pastel", ax=ax)
        ax.set_title(f"Distribution of {feature}")
        show_fig(fig)
    with right:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.boxplot(data=data, x="Target", y=feature, hue="Target",
                    palette="viridis", legend=False, ax=ax)
        ax.set_title(f"{feature} by outcome")
        show_fig(fig)


def page_correlation(pipe):
    st.title("🔗 Correlation Analysis")

    st.subheader("Correlation with the target")
    corr_target = (
        pipe["corr_matrix"][["Target"]].sort_values(by="Target", ascending=False)
    )
    left, right = st.columns([1, 1.4])
    with left:
        fig, ax = plt.subplots(figsize=(3.6, 11))
        sns.heatmap(corr_target, annot=True, fmt=".2f", cmap="coolwarm",
                    cbar=False, ax=ax)
        ax.set_title("Correlation with Target")
        show_fig(fig)
    with right:
        st.dataframe(
            corr_target.rename(columns={"Target": "Correlation"}).round(4),
            width="stretch",
            height=640,
        )
        st.caption(
            f"{pipe['positive'].shape[1] - 1} features have a positive correlation "
            "with graduation and are kept for modelling."
        )

    st.divider()
    st.subheader("Full correlation heatmap")
    show_full = st.checkbox("Render the full 35×35 heatmap (slow)", value=False)
    if show_full:
        fig, ax = plt.subplots(figsize=(24, 20))
        sns.heatmap(pipe["binary"].corr(), annot=True, fmt=".1f", cmap="coolwarm",
                    linewidths=0.5, annot_kws={"size": 6}, ax=ax)
        ax.set_title("Correlation Heatmap")
        show_fig(fig)
    else:
        fig, ax = plt.subplots(figsize=(14, 11))
        sns.heatmap(pipe["positive"].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                    linewidths=0.5, annot_kws={"size": 7}, ax=ax)
        ax.set_title("Correlation Heatmap — selected features")
        show_fig(fig)

    st.divider()
    st.subheader("Outlier removal (IQR, k = 10)")
    feature_cols = pipe["feature_cols"]
    before, after = pipe["positive"], pipe["clean"]
    st.caption(
        f"{before.shape[0]:,} rows → {after.shape[0]:,} rows "
        f"({before.shape[0] - after.shape[0]:,} removed)"
    )
    which = st.radio("Show", ["After removal", "Before removal"], horizontal=True)
    frame = after if which == "After removal" else before

    n = len(feature_cols)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(18, 2.4 * nrows))
    gs = GridSpec(nrows, ncols, figure=fig)
    for i, col in enumerate(feature_cols):
        ax = fig.add_subplot(gs[i])
        sns.boxplot(x="Target", y=col, data=frame, hue="Target",
                    palette="viridis", legend=False, ax=ax)
        ax.set_xlabel("Target", fontsize=8)
        ax.set_ylabel(col, fontsize=7)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    show_fig(fig)


def page_models(pipe):
    st.title("🤖 Model Training & Comparison")
    trained = train_models()
    results_df = trained["results_df"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training samples", f"{trained['train_shape'][0]:,}")
    c2.metric("Test samples", f"{trained['test_shape'][0]:,}")
    c3.metric("Features used", trained["train_shape"][1])
    c4.metric("Best C (Logistic)", trained["best_c"])
    st.caption(
        f"Training set — Dropout: {trained['train_counts'][0]:,} · "
        f"Graduate: {trained['train_counts'][1]:,}"
    )

    st.divider()
    st.subheader("Leaderboard")
    st.dataframe(
        results_df.style.format(
            {"Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}"}
        ).background_gradient(cmap="Greens", subset=["Accuracy", "Precision", "Recall", "F1"]),
        width="stretch",
        hide_index=True,
    )

    melted = results_df.melt(
        id_vars="Algorithm", var_name="Metric", value_name="Score"
    )
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.barplot(data=melted, x="Metric", y="Score", hue="Algorithm",
                palette="pastel", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Metric comparison across models")
    for container in ax.containers:
        if all(bar is not None for bar in container):
            ax.bar_label(container, fmt="%.3f", fontsize=8)
    show_fig(fig)

    st.divider()
    st.subheader("Per-model detail")
    tabs = st.tabs([r["Algorithm"] for r in trained["results"]])
    for tab, result in zip(tabs, trained["results"]):
        with tab:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{result['Accuracy']:.4f}")
            m2.metric("Precision", f"{result['Precision']:.4f}")
            m3.metric("Recall", f"{result['Recall']:.4f}")
            m4.metric("F1-score", f"{result['F1']:.4f}")

            left, right = st.columns([1, 1.1])
            with left:
                cm = confusion_matrix(trained["y_test"], result["y_pred"])
                fig, ax = plt.subplots(figsize=(5.5, 5))
                sns.heatmap(cm, fmt=".0f", cmap="YlGnBu", linewidth=1, square=True,
                            annot=True, annot_kws={"fontsize": 15},
                            xticklabels=["Dropout", "Graduate"],
                            yticklabels=["Dropout", "Graduate"], ax=ax)
                ax.set_xlabel("Prediction")
                ax.set_ylabel("Actual")
                ax.set_title(f"Confusion Matrix — {result['Algorithm']}")
                show_fig(fig)
            with right:
                report = classification_report(
                    trained["y_test"], result["y_pred"],
                    target_names=["Dropout", "Graduate"], output_dict=True,
                )
                st.dataframe(
                    pd.DataFrame(report).T.round(4), width="stretch"
                )

    st.divider()
    st.subheader("Feature influence — Logistic Regression")
    logistic = trained["models"]["Logistic Regression"]
    coefs = (
        pd.DataFrame(
            {"Feature": trained["feature_names"], "Coefficient": logistic.coef_[0]}
        )
        .sort_values("Coefficient", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(coefs) + 1.5))
    ordered = coefs.sort_values("Coefficient")
    colors = ["#e07a5f" if v < 0 else "#81b29a" for v in ordered["Coefficient"]]
    ax.barh(ordered["Feature"], ordered["Coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient")
    ax.set_title("Positive → pushes toward Graduate · Negative → pushes toward Dropout")
    show_fig(fig)


def page_predict(pipe):
    st.title("🔮 Predict a Student's Outcome")
    model, scaler, feature_names, source = get_predictor()
    st.caption(f"Using the Logistic Regression classifier ({source}).")

    reference = pipe["clean"]
    missing = [f for f in feature_names if f not in reference.columns]
    if missing:
        st.error(
            "These features are expected by the model but missing from the data: "
            + ", ".join(missing)
        )
        return

    tab_single, tab_batch = st.tabs(["Single student", "Batch (CSV upload)"])

    with tab_single:
        with st.sidebar:
            st.header("Prefill")
            preset = st.selectbox(
                "Start from",
                ["Median student", "A real Graduate", "A real Dropout"],
            )
            if preset == "Median student":
                defaults = reference[feature_names].median()
            elif preset == "A real Graduate":
                defaults = reference[reference["Target"] == 1][feature_names].sample(
                    1, random_state=st.session_state.get("seed", 1)
                ).iloc[0]
            else:
                defaults = reference[reference["Target"] == 0][feature_names].sample(
                    1, random_state=st.session_state.get("seed", 1)
                ).iloc[0]
            if st.button("Shuffle example", width="stretch"):
                st.session_state["seed"] = int(np.random.randint(0, 10_000))
                st.rerun()

        st.markdown("Adjust the inputs, then predict.")
        values = {}
        cols = st.columns(3)
        for i, feature in enumerate(feature_names):
            series = reference[feature]
            default = float(defaults[feature])
            with cols[i % 3]:
                uniques = np.sort(series.unique())
                is_categorical = len(uniques) <= 12 and np.all(
                    np.equal(np.mod(uniques, 1), 0)
                )
                if is_categorical:
                    options = [int(u) for u in uniques]
                    idx = options.index(int(default)) if int(default) in options else 0
                    values[feature] = st.selectbox(feature, options, index=idx)
                else:
                    step = 1.0 if series.dtype.kind in "iu" else 0.1
                    values[feature] = st.number_input(
                        feature,
                        min_value=float(series.min()),
                        max_value=float(series.max()),
                        value=default,
                        step=step,
                    )

        st.divider()
        if st.button("Predict outcome", type="primary", width="stretch"):
            row = pd.DataFrame([values])[feature_names]
            scaled = scaler.transform(row)
            pred = int(model.predict(scaled)[0])
            proba = model.predict_proba(scaled)[0]

            label = TARGET_NAMES[pred]
            confidence = proba[list(model.classes_).index(pred)]
            if pred == 1:
                st.success(f"### 🎓 Predicted: **{label}**")
            else:
                st.error(f"### ⚠️ Predicted: **{label}**")

            c1, c2, c3 = st.columns(3)
            c1.metric("Confidence", f"{confidence:.1%}")
            c2.metric("P(Dropout)", f"{proba[list(model.classes_).index(0)]:.1%}")
            c3.metric("P(Graduate)", f"{proba[list(model.classes_).index(1)]:.1%}")
            st.progress(float(proba[list(model.classes_).index(1)]))

            contrib = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Contribution": model.coef_[0] * scaled[0],
                }
            )
            contrib["abs"] = contrib["Contribution"].abs()
            top = contrib.nlargest(12, "abs").sort_values("Contribution")
            fig, ax = plt.subplots(figsize=(9, 5))
            colors = ["#e07a5f" if v < 0 else "#81b29a" for v in top["Contribution"]]
            ax.barh(top["Feature"], top["Contribution"], color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title("Top drivers of this prediction")
            ax.set_xlabel("← toward Dropout        toward Graduate →")
            show_fig(fig)

    with tab_batch:
        st.markdown(
            "Upload a CSV containing the model's feature columns. "
            "Extra columns are ignored."
        )
        with st.expander("Required columns"):
            st.code("\n".join(feature_names))
            template = reference[feature_names].head(5)
            st.download_button(
                "Download a template CSV",
                template.to_csv(index=False).encode(),
                file_name="prediction_template.csv",
                mime="text/csv",
            )

        uploaded = st.file_uploader("CSV file", type=["csv"])
        if uploaded is not None:
            batch = pd.read_csv(uploaded, encoding="utf-8-sig")
            absent = [f for f in feature_names if f not in batch.columns]
            if absent:
                st.error("Missing required columns: " + ", ".join(absent))
            else:
                scaled = scaler.transform(batch[feature_names])
                preds = model.predict(scaled)
                proba = model.predict_proba(scaled)
                graduate_idx = list(model.classes_).index(1)

                out = batch.copy()
                out["Prediction"] = [TARGET_NAMES[int(p)] for p in preds]
                out["P(Graduate)"] = proba[:, graduate_idx].round(4)
                out["P(Dropout)"] = (1 - proba[:, graduate_idx]).round(4)

                c1, c2, c3 = st.columns(3)
                c1.metric("Rows scored", f"{len(out):,}")
                c2.metric("Predicted dropouts", int((preds == 0).sum()))
                c3.metric("Predicted graduates", int((preds == 1).sum()))

                st.dataframe(out, width="stretch", height=420)
                st.download_button(
                    "Download predictions",
                    out.to_csv(index=False).encode(),
                    file_name="predictions.csv",
                    mime="text/csv",
                    type="primary",
                )

def main():
    if not os.path.exists(DATA_PATH):
        st.error(f"`data.csv` not found at {DATA_PATH}")
        st.stop()

    pipe = build_pipeline()

    st.sidebar.title("🎓 Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Overview", "Exploratory Analysis", "Correlation", "Models", "Predict"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()

    {
        "Overview": page_overview,
        "Exploratory Analysis": page_eda,
        "Correlation": page_correlation,
        "Models": page_models,
        "Predict": page_predict,
    }[page](pipe)

    st.sidebar.caption(
        "Dataset: Predict Students' Dropout and Academic Success (UCI). "
        "Pipeline mirrors `Untitled.ipynb`."
    )


if __name__ == "__main__":
    main()