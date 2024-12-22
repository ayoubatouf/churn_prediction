from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from config.config import EDA_PATH, RAW_DATA_PATH


def plot_monthly_charges_distribution(data: pd.DataFrame, base_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    sns.histplot(data["MonthlyCharges"], bins=30, kde=True)
    plt.title("Distribution of Monthly Charges")
    plt.xlabel("Monthly Charges")
    plt.ylabel("Frequency")
    plt.savefig(
        os.path.join(base_path, "monthly_charges_distribution.png"), bbox_inches="tight"
    )
    plt.close()


def plot_churn_rate_by_gender(data: pd.DataFrame, base_path: Path) -> None:
    churn_gender = (
        data.groupby("gender")["Churn"].value_counts(normalize=True).unstack()
    )
    churn_gender.plot(kind="bar", stacked=True, figsize=(8, 5))
    plt.title("Churn Rate by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Proportion")
    plt.legend(title="Churn", labels=["No", "Yes"])
    plt.savefig(
        os.path.join(base_path, "churn_rate_by_gender.png"), bbox_inches="tight"
    )
    plt.close()


def plot_churn_rate_by_contract(data: pd.DataFrame, base_path: Path) -> None:
    churn_contract = (
        data.groupby("Contract")["Churn"].value_counts(normalize=True).unstack()
    )
    churn_contract.plot(kind="bar", stacked=True, figsize=(8, 5))
    plt.title("Churn Rate by Contract Type")
    plt.xlabel("Contract Type")
    plt.ylabel("Proportion")
    plt.legend(title="Churn", labels=["No", "Yes"])
    plt.savefig(
        os.path.join(base_path, "churn_rate_by_contract.png"), bbox_inches="tight"
    )
    plt.close()


def plot_average_monthly_charges_by_internet_service(
    data: pd.DataFrame, base_path: Path
) -> None:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="InternetService", y="MonthlyCharges", data=data)
    plt.title("Average Monthly Charges by Internet Service")
    plt.xlabel("Internet Service")
    plt.ylabel("Monthly Charges")
    plt.savefig(
        os.path.join(base_path, "average_monthly_charges_by_internet_service.png"),
        bbox_inches="tight",
    )
    plt.close()


def plot_churn_rate_by_senior_citizen_status(
    data: pd.DataFrame, base_path: Path
) -> None:
    churn_senior = (
        data.groupby("SeniorCitizen")["Churn"].value_counts(normalize=True).unstack()
    )
    churn_senior.plot(kind="bar", stacked=True, figsize=(6, 4))
    plt.title("Churn Rate by Senior Citizen Status")
    plt.xlabel("Senior Citizen (0 = No, 1 = Yes)")
    plt.ylabel("Proportion")
    plt.legend(title="Churn", labels=["No", "Yes"])
    plt.savefig(
        os.path.join(base_path, "churn_rate_by_senior_citizen.png"), bbox_inches="tight"
    )
    plt.close()


if __name__ == "__main__":

    raw_data_path = RAW_DATA_PATH
    data = pd.read_csv(raw_data_path)
    base_path = EDA_PATH

    plot_monthly_charges_distribution(data, base_path)
    plot_churn_rate_by_gender(data, base_path)
    plot_churn_rate_by_contract(data, base_path)
    plot_average_monthly_charges_by_internet_service(data, base_path)
    plot_churn_rate_by_senior_citizen_status(data, base_path)
    print("finishing plotting !")
