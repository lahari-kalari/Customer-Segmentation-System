
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Locate the customer dataset
data_path = Path(__file__).parent / "data" / "customers.csv"


def main():
    # Check whether the dataset exists
    if not data_path.exists():
        print("Error: customers.csv not found in the data folder.")
        return

    # Load the dataset
    data = pd.read_csv(data_path)

    required_columns = {"CustomerID", "AnnualIncome", "SpendingScore"}
    if not required_columns.issubset(data.columns):
        print("Error: The dataset is missing required columns.")
        return

    if data[list(required_columns)].isnull().any().any():
        print("Error: Please remove missing values from the dataset.")
        return

    # Select features for customer segmentation
    features = data[["AnnualIncome", "SpendingScore"]]

    # Scale the features so income does not dominate the model
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Train the K-Means clustering model
    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    data["Cluster"] = model.fit_predict(scaled_features)

    print("=== Customer Segmentation System ===")
    print("Model trained successfully!")

    print("\nCustomer groups:")
    print(data[["CustomerID", "AnnualIncome", "SpendingScore", "Cluster"]]
          .to_string(index=False))

    print("\nNumber of customers in each cluster:")
    print(data["Cluster"].value_counts().sort_index().to_string())

    # Predict the cluster for a new customer
    try:
        income = float(input("\nEnter annual income (same units as dataset): "))
        spending = float(input("Enter spending score (0-100): "))

        if income < 0:
            print("Annual income cannot be negative.")
            return

        if not 0 <= spending <= 100:
            print("Spending score must be between 0 and 100.")
            return

        new_customer = pd.DataFrame({
            "AnnualIncome": [income],
            "SpendingScore": [spending]
        })

        scaled_customer = scaler.transform(new_customer)
        predicted_cluster = model.predict(scaled_customer)[0]

        print(f"\nThe new customer belongs to Cluster {predicted_cluster}.")

        # Plot the customer groups
        plt.figure(figsize=(8, 5))

        for cluster in sorted(data["Cluster"].unique()):
            group = data[data["Cluster"] == cluster]
            plt.scatter(
                group["AnnualIncome"],
                group["SpendingScore"],
                label=f"Cluster {cluster}"
            )

        plt.scatter(
            income,
            spending,
            marker="*",
            s=250,
            label="New Customer"
        )

        plt.xlabel("Annual Income")
        plt.ylabel("Spending Score")
        plt.title("Customer Segmentation using K-Means")
        plt.legend()
        plt.tight_layout()
        plt.show()

    except ValueError:
        print("Please enter valid numeric values.")


if __name__ == "__main__":
    main()
