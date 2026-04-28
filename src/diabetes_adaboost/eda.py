"""Exploratory plots from the original notebook (factored into functions)."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns


def _finish_plot(save_path: Optional[str] = None) -> None:
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


def hist_insulin(df, save_path: Optional[str] = None):
    sns.histplot(df["Insulin"], bins=30, kde=True)
    _finish_plot(save_path)


def hist_pedigree(df, save_path: Optional[str] = None):
    sns.histplot(df["DiabetesPedigreeFunction"], bins=30, kde=True, color="red")
    _finish_plot(save_path)


def heatmap_correlation(df, save_path: Optional[str] = None):
    sns.heatmap(data=df.corr(), cmap="coolwarm", annot=True)
    _finish_plot(save_path)


def boxplot_glucose_by_outcome(df, save_path: Optional[str] = None):
    sns.boxplot(data=df, x="Outcome", y="Glucose")
    _finish_plot(save_path)


def boxplot_insulin_by_outcome(df, save_path: Optional[str] = None):
    sns.boxplot(data=df, x="Outcome", y="Insulin")
    _finish_plot(save_path)


def boxplot_numeric_features(df, save_path: Optional[str] = None):
    sns.boxplot(data=df[["Glucose", "BloodPressure", "SkinThickness", "BMI"]])
    plt.title("Box Plot of Numerical Features")
    _finish_plot(save_path)


def scatter_glucose_age(df, save_path: Optional[str] = None):
    sns.scatterplot(
        data=df, x="Age", y="Glucose", hue="Outcome", size="Pregnancies"
    )
    plt.title("Glucose and Age")
    _finish_plot(save_path)


def scatter_glucose_blood_pressure(df, save_path: Optional[str] = None):
    sns.scatterplot(x="Glucose", y="BloodPressure", data=df, hue="Outcome")
    plt.title("Glucose vs Blood Pressure")
    _finish_plot(save_path)


def scatter_glucose_insulin(df, save_path: Optional[str] = None):
    sns.scatterplot(data=df, x="Glucose", y="Insulin", hue="Outcome")
    _finish_plot(save_path)


def line_insulin_pedigree(df, save_path: Optional[str] = None):
    sns.lineplot(data=df, x="Insulin", y="DiabetesPedigreeFunction")
    _finish_plot(save_path)


def pairplot_subset(df, save_path: Optional[str] = None):
    g = sns.pairplot(
        df[["Insulin", "BloodPressure", "SkinThickness", "Glucose", "BMI", "Age"]]
    )
    if save_path:
        g.savefig(save_path, bbox_inches="tight", dpi=120)
        plt.close("all")
    else:
        plt.show()


def stripplot_age_outcome(df, save_path: Optional[str] = None):
    sns.stripplot(x="Outcome", y="Age", data=df, jitter=True, alpha=0.6)
    _finish_plot(save_path)
