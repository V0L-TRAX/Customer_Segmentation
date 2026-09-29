# Customer Segmentation Analysis

## 📌 Project Title

**End-to-End Customer Segmentation and Clustering Analysis in Python**

---

## 🎯 Project Objective

The primary objective of this project is to build a customer segmentation system using K-Means clustering. The project demonstrates how customer characteristics such as age, gender, annual income, and spending score can be analyzed and used to group similar customers into meaningful segments.

---

## ⚠️ Problem Statement

Customers have different demographic characteristics, income levels, and spending behavior. Treating all customers in the same way can make it difficult to understand their needs and behavior.

This project addresses this challenge by applying exploratory analysis, feature scaling, cluster selection, K-Means clustering, visualization, and business interpretation to identify groups of similar customers.

---

## 📊 Dataset Description

The project uses the **classic Mall Customers dataset**, containing **200 customer records**.

### Dataset Columns

- `CustomerID` — unique customer identifier
- `Gender` — customer gender
- `Age` — customer age
- `Annual Income (k$)` — annual income
- `Spending Score (1-100)` — spending score

---

## 🔧 Project Workflow

The customer segmentation pipeline follows these major steps:

1. Load the customer dataset
2. Perform Exploratory Data Analysis
3. Select relevant segmentation features
4. Scale the selected features
5. Determine the appropriate number of clusters
6. Apply the Elbow Method and Silhouette Score
7. Train the K-Means clustering model
8. Assign cluster labels to customers
9. Visualize the resulting customer segments
10. Analyze and interpret each segment

---

## 🤖 Clustering Method

The project uses **K-Means clustering** to group customers according to their characteristics and spending behavior.

The segmentation primarily uses:

- Annual Income
- Spending Score

Feature scaling is performed before clustering so that the selected variables can contribute appropriately to the clustering process.

---

## 📈 Cluster Analysis

The project includes:

- Elbow Method analysis
- Silhouette Score analysis
- Customer cluster visualization
- Cluster box plots
- Cluster-level statistics
- Business interpretation of customer segments

The resulting clustered dataset is saved as:

`data/customers_with_clusters.csv`

---

## 📊 Visualizations

The project generates:

1. `elbow_silhouette.png` — Elbow and Silhouette analysis
2. `customer_clusters.png` — Main customer cluster scatter plot
3. `cluster_boxplots.png` — Feature distributions by cluster

All visualizations are saved in the `images/` folder.

---

## 📁 Project Structure

```text
customer_segmentation/
├── data/
│   ├── Mall_Customers.csv
│   └── customers_with_clusters.csv
├── images/
│   ├── elbow_silhouette.png
│   ├── customer_clusters.png
│   └── cluster_boxplots.png
├── customer_segmentation.py
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

### Run with Python

Open a terminal in the `customer_segmentation` folder.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the segmentation analysis:

```bash
python customer_segmentation.py
```

The script will:

- Print analysis results in the terminal
- Save visualization images in the `images/` folder
- Save the clustered dataset in `data/customers_with_clusters.csv`

### Run with Jupyter Notebook / Google Colab

Install Jupyter if required:

```bash
pip install notebook
```

Start Jupyter:

```bash
python -m notebook
```

Open the included notebook and select:

**Kernel → Restart Kernel and Run All**

For Google Colab, upload the dataset and notebook, then make sure the CSV path is correct.

---

## 📌 Expected Output Segments

Typical segments produced from this dataset can include:

| Income | Spending | Example Segment |
|---|---|---|
| High | High | Premium / VIP |
| Low | High | High Spenders / Low Income |
| High | Low | Potential / Target |
| Low | Low | Budget / Low Value |
| Medium | Medium | Average / Standard |

These segment descriptions are used to help interpret the clustering results and connect them with possible business actions.

---

## 📦 Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

---

## 📝 Conclusion

This project demonstrates how businesses can use exploratory analysis and K-Means clustering to identify groups of similar customers, visualize their characteristics, and interpret customer behavior for data-driven segmentation.
