#  Customer Segmentation Analysis Model

## Project Overview
This project creates a **customer segmentation system** using **K-Means clustering**.  
It groups mall customers based on **Annual Income** and **Spending Score**, then visualizes and explains the resulting segments.

---

## What This Project Does
1. Loads customer data (Age, Gender, Income, Spending Score)
2. Performs Exploratory Data Analysis (EDA)
3. Scales the features
4. Finds the optimal number of clusters (Elbow Method + Silhouette Score)
5. Trains a K-Means model
6. Visualizes the customer segments
7. Provides business interpretation for each segment

---

## Project Structure
```
customer_segmentation/
├── data/
│   ├── Mall_Customers.csv              # Original dataset
│   └── customers_with_clusters.csv     # Output with cluster labels
├── images/
│   ├── elbow_silhouette.png            # Elbow + Silhouette plots
│   ├── customer_clusters.png           # Main cluster scatter plot
│   └── cluster_boxplots.png            # Boxplots by cluster
├── customer_segmentation.py            # Main Python script
├── requirements.txt
└── README.md
```

---

## How to Run

### Option 1: Run the Python Script (Recommended)

1. Open terminal in the `customer_segmentation` folder
2. Install dependencies (only needed once):
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python customer_segmentation.py
   ```

The script will:
- Print analysis results in the terminal
- Save 3 visualization images in the `images/` folder
- Save the clustered dataset in `data/customers_with_clusters.csv`

### Option 2: Run in Google Colab / Jupyter

1. Upload `Mall_Customers.csv` and the script
2. Or copy the code into a notebook cell by cell
3. Make sure the path to the CSV is correct

---

## Dataset
- **Source**: Classic Mall Customers dataset (200 customers)
- **Columns**:
  - `CustomerID`
  - `Gender`
  - `Age`
  - `Annual Income (k$)`
  - `Spending Score (1-100)`

---

## Expected Output Segments (typical for this dataset)

| Cluster | Income     | Spending   | Segment Name                          | Business Action                        |
|---------|------------|------------|---------------------------------------|----------------------------------------|
| High    | High       | High       | Premium / VIP                         | Loyalty programs, exclusive offers     |
| Low     | High       | High       | Impulse / High Spenders (Low Income)  | Discounts & value deals                |
| High    | Low        | Low        | Target / Potential                    | Upselling campaigns                    |
| Low     | Low        | Low        | Budget / Low Value                    | Cost-effective retention               |
| Medium  | Medium     | Medium     | Average / Standard                    | Standard marketing                     |

---

## Requirements
- Python 3.8+
- pandas, numpy, scikit-learn, matplotlib, seaborn

---

## Author
Created for **IncodeVision Task 04 – Customer Segmentation Analysis**
