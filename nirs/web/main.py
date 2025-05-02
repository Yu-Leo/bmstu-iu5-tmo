import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


def load_data():
    return pd.read_csv("../dataset.csv")


def model_training_page():
    st.title("Обучение модели RandomForest для определения кредитного рейтинга")

    df = load_data()

    X = df.drop(columns=["credit_score"])
    y = df["credit_score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    st.subheader("Параметры модели")

    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider("Количество деревьев (n_estimators)", 10, 1000, 100, step=10)
    with col2:
        max_depth = st.selectbox("Максимальная глубина дерева (max_depth)", [None, 1, 10, 100, 1000])

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=1)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    metrics_data = {
        "Метрика": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Значение": [f"{accuracy * 100:.2f}%", f"{precision * 100:.2f}%", f"{recall * 100:.2f}%", f"{f1 * 100:.2f}%"]
    }
    metrics_df = pd.DataFrame(metrics_data)

    st.subheader("Метрики модели")
    st.table(metrics_df.set_index("Метрика"))

    cm = confusion_matrix(y_test, y_pred)
    st.subheader("Матрица ошибок")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Poor", "Standard", "Good"], yticklabels=["Poor", "Standard", "Good"])
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Реальное значение")
    st.pyplot(fig)


def main():
    model_training_page()

if __name__ == "__main__":
    main()

