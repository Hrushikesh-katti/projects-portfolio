
# 🎬 Movie Success Prediction & Marketing AI

## 📌 Project Overview

Welcome to **Movie Success Prediction & Marketing AI**, a machine learning project designed to predict whether a movie will be a **Flop**, **Average**, or **Hit** based on historical movie metadata, and generate tailored marketing taglines and strategies using a **large language model (LLM)**.

This project showcases a complete data science pipeline—from **EDA**, **data cleaning**, **feature engineering**, and **A/B model testing**, to **hyperparameter optimization** and **deployment** via a Gradio interface.

By leveraging **Python**, **Scikit-learn**, **XGBoost**, and **DeepSeek (via OpenRouter)**, I built a robust system that not only predicts movie success but also generates creative marketing solutions.

---

## 📊 Dataset

- **File**: `movie_metadata.csv`
- **Size**: ~1.42 MB
- **Samples**: 5,000+
- **Source**: Kaggle

### Key Features:
- **Numerical**: Budget, gross, IMDb score, user/critic reviews, social media engagement
- **Categorical**: Genres, content rating, language, country
- **Target**: IMDb score converted into categories:
  - **Flop**: < 5.5
  - **Average**: 5.5 – 7.0
  - **Hit**: > 7.0

➡️ Full feature descriptions available in [`dataset/data_description.txt`](./dataset/data_description.txt)

---

## 🧠 Methodology

### 1️⃣ Exploratory Data Analysis (EDA)
- Checked missing values, outliers, skewness
- Visualized with histograms, boxplots, heatmaps
- Noted class imbalance: **Hit (1104)**, **Average (1798)**, **Flop (596)**
- Chose class-weighted models to compensate

### 2️⃣ Data Cleaning
- Dropped duplicates
- Imputed missing values with median/mode
- Filled `plot_keywords` with `"unknown"` for LLM compatibility
- Applied log transformations and clipping on skewed values

### 3️⃣ Feature Engineering & A/B Testing
Created two datasets:
- **df1**: Original features
- **df2**: Feature-engineered with weighted engagement score

New Features:
- `budget_to_gross_ratio`
- `genre_count`
- `engagement_score` (from user votes & reviews)

Removed irrelevant columns (e.g., actor names) to reduce noise.

### 4️⃣ Model Training & Evaluation
Models used:
- RandomForest
- GradientBoosting
- XGBoost

- Encoding: **One-hot** & **label** encoding for *categoricals*, **MultiLabelBinarizer** for `genres`
- Scaling: **StandardScaler**
- Tuning: **GridSearchCV** (on 50% of data)
- **Handling Class Imbalance**: Used `class_weight='balanced'` in RandomForest to auto-adjust weights for minority classes (Flop).  
- **Alternatives Considered**: SMOTE was skipped due to computational overhead with tree-based models.  
- Metrics: Macro F1-score, accuracy, per-class F1

#### 🏆 Results:
| Dataset | Model         | Macro F1 |
|---------|---------------|----------|
| df1     | RandomForest  | 0.681 ✅ |
| df1     | GradientBoost | 0.680    |
| df1     | XGBoost       | 0.677    |
| df2     | RF / GB / XGB | ~0.670   |

- Per-class performance: F1-scores were highest for **Average** and **Hit**, lower for **Flop**
- **Top 3 Features**: `num_voted_users`, `duration`, `num_critic_for_reviews` (from RandomForest on df1)

### 5️⃣ LLM-Driven Tagline & Strategy Generator
Integrated **DeepSeek-v3 (via OpenRouter API)** to:
- Generate 5 unique taglines for a movie based on `genres` and `plot_keywords`
- Suggest marketing strategies based on predicted outcome (Hit/Average/Flop), `genres` and `plot_keywords`

### 6️⃣ Deployment Optimization
- Original notebook runtime: **6.3 minutes**
- Deployment script runtime: **9.2 seconds** ✅

Optimizations included:
- Removing model comparisons
- Using pre-determined best hyperparameters from **GridSearchCV**
- Deploying via **Gradio** interface

---

## 📈 Results Summary

- **Best Model**: RandomForest on df1 (macro F1: **0.681**)
- **Optimization**: Runtime reduced by **98.5%**
- **Top Features**: `num_voted_users`, `duration`, `num_critic_for_reviews`
- **LLM Integration**: Generated real-time taglines and strategies
- **Saved Visualizations**:
  - `macro_f1_comparison.png`
  - `per_class_f1_comparison.png`
  - `feature_importance_df1.png`

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/hrushikesh-katti-portfolio/movie-success-marketing-ai.git
cd movie-success-marketing-ai
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
Ensure Python 3.11.8 is installed, then run:

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset
Place `movie_metadata.csv` in the `dataset/` folder.

---

### 5. Run the Notebook
```bash
jupyter notebook MovieSuccessML-and-MarketingAI.ipynb
```

### 6. Launch the Deployment App
```bash
python MovieSuccess-AI-app.py
```

> ⚠️ **Note**: Replace the OpenRouter API key in the scripts with your own from [openrouter.ai](https://openrouter.ai)

---

## 🧾 Project Structure

```
movie-success-marketing-ai/
├── MovieSuccessML-and-MarketingAI.ipynb
├── MovieSuccess-AI-app.py
├── requirements.txt
├── README.md
├── dataset/
│   ├── movie_metadata.csv
│   └── data_description.txt
├── macro_f1_comparison.png
├── per_class_f1_comparison.png
└── feature_importance_df1.png
```

---

## 🏆 Key Achievements

- ✅ **Full end-to-end ML pipeline**
- ✅ A/B testing of features (original vs. engineered)
- ✅ Optimized deployment (from 6.3 mins → 9.2 secs)
- ✅ Integrated LLM for content generation
- ✅ Professionally structured, documented, and ready for production

---

## 🚀 Future Improvements

- Use **SMOTE** to improve Flop class prediction
- Enhance LLM prompts or fine-tune for more tailored taglines
- Extract sentiment/clusters from `plot_keywords` using NLP
- Containerize the app with **Docker** for scalable deployment

---

## 👨‍💻 About Me


**Hrushikesh S Katti**   
**Data Science Specialist | AI Certified Professional | Machine Learning Engineer | NLP Expert | Business Intelligence Analyst | Marketing Strategist | Deep Learning Practitioner | Python for Data Science | Prompt Engineer**  
🔍 Open to Work | Actively Looking for Full-Time Opportunities
📍 Open to Relocation: Bangalore | Hyderabad | Pune | Chennai | Remote  

- GitHub: [Hrushikesh-katti](https://github.com/Hrushikesh-katti)  
- Email: hrushikeshskatti7@gmail.com  

---

> ✨ *Thank you for checking out my project! What started as code became a journey – may it challenge and inspire you as much as it did me.* 🚀  
> *From BBA to AI-ML: Proving data doesn't care about your degree, just your dedication.* 💡
