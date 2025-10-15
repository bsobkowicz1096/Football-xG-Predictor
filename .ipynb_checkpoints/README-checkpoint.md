# ⚽ Football Expected Goals (xG) Predictor

## 🧠 About the Project

An Expected Goals (xG) model that predicts the probability of scoring based on StatsBomb data. The project uses machine learning techniques to analyze factors most influencing shot effectiveness in football. Applied models (Logistic Regression, Random Forest, XGBoost) combined with Beta calibration technique create a highly accurate predictive tool. Analysis results confirm the crucial role of shot geometry and defender influence on goal-scoring probability.

## 🎯 Motivation

Expected Goals (xG) is one of the most important metrics used in modern football analytics. It allows for evaluating shot quality regardless of whether they resulted in a goal. In this project, I built my own xG model to better understand factors affecting shot effectiveness and create a tool that can be used for match analysis and player evaluation.

## 📋 Data

The data used comes from StatsBomb's open dataset from the 2015/2016 season for five top European leagues:
- Premier League (England)
- La Liga (Spain)
- Bundesliga (Germany)
- Serie A (Italy)
- Ligue 1 (France)

The data contains detailed information about each shot, including position on the pitch, shot type, circumstances of the shot, and positioning of other players at the moment of the shot.

https://github.com/statsbomb/open-data

## 🔍 Methodology

### Data Preparation
- Extraction of relevant shot-related variables
- Transformation of raw location data into useful geometric features
- Categorization of shot types and body parts used for shots

### Feature Engineering
- **Geometric**: shot angle, distance from goal
- **Contextual**: number of defenders on shot line, goalkeeper presence
- **Technical**: dominant vs non-dominant foot shots, first-time shots
- **Situational**: shots under pressure, shots after dribbling

### Modeling
Testing and comparison of three algorithms:
1. Logistic Regression
2. Random Forest
3. XGBoost

### Model Calibration
Application of Beta Calibration technique to calibrate probabilities, which significantly improved model prediction quality.

## 📈 Key Results

### Model Comparison
| Model               | ROC AUC | Brier Score | Log Loss | xG/Goals Ratio |
|---------------------|---------|-------------|----------|----------------|
| Logistic Regression | 0.796   | 0.073       | 0.257    | 0.98           |
| Random Forest       | 0.796   | 0.074       | 0.259    | 0.99           |
| XGBoost             | 0.798   | 0.073       | 0.257    | 0.98           |

### Key Findings
1. **Shot geometry** is crucial - shot angle and distance from goal are the strongest predictors
2. **Defenders on shot line** - each additional defender significantly decreases goal-scoring probability
3. **First-time shots** have higher effectiveness than those preceded by ball control
4. **Model calibration** is crucial - all models before calibration significantly overestimated probabilities

## 💻 Technologies

- **Language**: Python 3.7+
- **Data Analysis**: Pandas, NumPy
- **ML Models**: Scikit-learn, XGBoost
- **Visualization**: Matplotlib, Seaborn, Mplsoccer
- **Data Source**: StatsBombPy

## 📁 Project Structure
```
Football-xG-Predictor/
├── notebooks/                 
│   ├── data_collection.py      # Data collection script
│   └── xg_model.ipynb          # Main notebook with xG model
├── src/                        
│   ├── __init__.py             # Package initialization file
│   ├── preprocessing.py        # Data preprocessing functions
│   ├── feature_engineering.py  # Feature engineering
│   ├── modeling.py             # Model implementation
│   ├── evaluation.py           # Metrics and model evaluation
│   └── visualization.py        # Visualizations
├── data/                       # Data folder
├── assets/                     # Graphics and visualizations
├── requirements.txt            # Dependencies
└── README.md                   # Project description / this file
```

## 🚀 How to Download and Run the Project

1. Clone the repository:
   
```bash
git clone https://github.com/bsobkowicz1096/Football-xG-Predictor.git
```
2. Navigate to the project directory:

```bash
cd Football-xG-Predictor
```
3. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate     # On Windows
```
4. Install dependencies:
```bash
pip install -r requirements.txt
```
5. Run the notebook:
```bash
jupyter notebook notebooks/football_xg_predictor.ipynb
```

Note: The project uses publicly available StatsBomb data, used in accordance with their license terms.