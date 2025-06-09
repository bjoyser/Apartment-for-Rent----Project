# 🏢 Apartment Rental Market Segmentation Using K-Means Clustering

## 🎯 Project Objective
This project applies unsupervised machine learning to segment apartment rental listings into meaningful market categories. The goal is to help real estate platforms, investors, and renters understand pricing patterns and property features across different market segments.

## 📊 Dataset Overview
- **Dataset Name**: `apartment for rent dataset.csv`
- **Rows**: ~100,000 listings
- **Key Features**: Price, bedrooms, bathrooms, square footage, amenities (pool/gym/parking), location
- **Source**: [https://archive.ics.uci.edu/dataset/555/apartment+for+rent+classified]

## 🛠️ Tools & Libraries Used
- **Python** (Pandas, NumPy)
- **Scikit-learn** (KMeans, StandardScaler, PCA, Silhouette Score)
- **Seaborn & Matplotlib** (Visualizations)
- **Joblib** (Model Persistence)
- **Jupyter Notebook**

## 🔍 Key Steps Performed

### 1. Data Cleaning & Preprocessing
- Handled missing values (amenities, square footage)
- Removed duplicate listings
- Converted price to numeric format
- Created new features (price per sqft, amenity flags)

### 2. Exploratory Data Analysis (EDA)
- Price distribution analysis
- Bedroom/price relationships
- City-wise median prices (Top 20 cities)
- Amenity availability trends

### 3. Outlier Detection & Treatment
- Identified outliers using IQR method
- Visualized outliers for key features
- Applied capping to extreme values

### 4. Feature Engineering & Scaling
- Selected 8 key features for clustering
- Standardized features using StandardScaler
- Correlation analysis of features

### 5. Market Segmentation Using K-Means
- Determined optimal k=4 using Elbow Method (SSE=1.2M) and Silhouette Score (0.42)
- Performed PCA for 2D cluster visualization

### 6. Cluster Profiling & Interpretation
- Analyzed median values for each cluster
- Assigned descriptive labels based on characteristics

## 📈 Market Insights

| Cluster | Label                 | Key Characteristics                          | Business Implications                     |
|---------|-----------------------|---------------------------------------------|------------------------------------------|
| 0       | Budget Apartments     | Low price, small size, basic amenities      | Target students/young professionals      |
| 1       | Luxury Apartments     | High price, large size, premium amenities   | Focus on affluent renters                |
| 2       | Mid-Range Family Homes| Moderate price, 3+ bedrooms                 | Ideal for family-oriented marketing      |
| 3       | Small Premium Units   | High price/sqft, compact, good amenities    | Target urban professionals               |

## 📂 Outputs
- **Trained Models**: `apartment_kmeans_model.pkl`, `apartment_scaler.pkl`
- **Visualizations**: All key plots as .png files
- **Jupyter Notebook**: Complete analysis with executable code

## 📎 How to Run
1. Clone this repository
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn joblib
