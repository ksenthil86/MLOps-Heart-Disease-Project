import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")


def plot_class_balance(df: pd.DataFrame, target_col: str = "target") -> None:
    """Plots class balance of the target variable."""
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(
        data=df,
        x=target_col
    )

    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha="center", va="bottom"
        )

    plt.title("Target Class Distribution")
    plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_numeric_distributions(df: pd.DataFrame,
                               numeric_features: list[str]) -> None:
    """Plots histograms for numeric features."""
    n_features = len(numeric_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    for idx, col in enumerate(numeric_features, start=1):
        plt.subplot(n_rows, n_cols, idx)
        sns.histplot(
            data=df,
            x=col,
            kde=True,
            bins=30
        )
        plt.title(f"Distribution of {col}")

    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame,
                             numeric_features: list[str]) -> None:
    """Plots correlation heatmap for numeric features."""
    corr = df[numeric_features + ["target"]].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.show()


def run_eda(df: pd.DataFrame) -> None:
    """Runs full EDA suite."""
    numeric_features = [
        'age', 'trestbps', 'chol', 'thalach', 'oldpeak'
    ]

    plot_class_balance(df)
    plot_numeric_distributions(df, numeric_features)
    plot_correlation_heatmap(df, numeric_features)
