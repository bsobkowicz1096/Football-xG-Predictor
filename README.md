# ⚽ Football Expected Goals (xG) Predictor

## 🧠 O projekcie

Model Expected Goals (xG) przewidujący prawdopodobieństwo strzelenia bramki na podstawie danych z StatsBomb. Projekt wykorzystuje techniki uczenia maszynowego do analizy czynników najbardziej wpływających na skuteczność strzałów w piłce nożnej. Zastosowane modele (Regresja Logistyczna, Random Forest, XGBoost) wraz z techniką kalibracji Beta tworzą narzędzie o wysokiej dokładności predykcyjnej. Wyniki analizy potwierdzają kluczową rolę geometrii strzału oraz wpływu obrońców na prawdopodobieństwo zdobycia bramki.

## 🎯 Motywacja

Expected Goals (xG) to jedna z najważniejszych miar stosowanych we współczesnej analizie piłkarskiej. Pozwala ona na ocenę jakości sytuacji strzeleckich niezależnie od tego, czy zakończyły się one bramką. W tym projekcie zbudowałem własny model xG, aby lepiej zrozumieć czynniki wpływające na skuteczność strzałów oraz stworzyć narzędzie, które może służyć do analizy meczów i oceny zawodników.

## 📋 Dane

Wykorzystane dane pochodzą z ogólnodostępnego zbioru StatsBomb z sezonu 2015/2016 dla pięciu czołowych lig europejskich:
- Premier League (Anglia)
- La Liga (Hiszpania)
- Bundesliga (Niemcy)
- Serie A (Włochy)
- Ligue 1 (Francja)

Dane zawierają szczegółowe informacje o każdym strzale, w tym pozycję na boisku, typ strzału, okoliczności jego oddania oraz ustawienie innych zawodników w momencie strzału.

https://github.com/statsbomb/open-data

## 🔍 Metodologia

### Przygotowanie danych
- Ekstrakcja istotnych zmiennych związanych ze strzałami
- Przekształcenie surowych danych lokalizacyjnych na użyteczne cechy geometryczne
- Kategoryzacja typów strzałów i części ciała użytych do ich oddania

### Inżynieria cech
- **Geometryczne**: kąt strzału, odległość od bramki
- **Kontekstowe**: liczba obrońców na linii strzału, obecność bramkarza
- **Techniczne**: strzały nogą dominującą vs niedominującą, strzały z pierwszej piłki
- **Sytuacyjne**: strzały pod presją, strzały po dryblingu

### Modelowanie
Testowanie i porównanie trzech algorytmów:
1. Regresja Logistyczna
2. Random Forest
3. XGBoost

### Kalibracja modelu
Zastosowanie techniki Beta Calibration do kalibracji prawdopodobieństw, co znacząco poprawiło jakość predykcji modelu.

## 📈 Kluczowe wyniki

### Porównanie modeli
| Model               | ROC AUC | Brier Score | Log Loss | xG/Goals Ratio |
|---------------------|---------|-------------|----------|----------------|
| Regresja Logistyczna| 0.796   | 0.073       | 0.257    | 0.98           |
| Random Forest       | 0.796   | 0.074       | 0.259    | 0.99           |
| XGBoost             | 0.798   | 0.073       | 0.257    | 0.98           |

### Najważniejsze odkrycia
1. **Geometria strzału** ma kluczowe znaczenie - kąt strzału i odległość od bramki to najsilniejsze predyktory
2. **Obrońcy na linii strzału** - każdy dodatkowy obrońca znacząco zmniejsza prawdopodobieństwo zdobycia bramki
3. **Strzały z pierwszej piłki** mają wyższą skuteczność niż te poprzedzone przyjęciem
4. **Kalibracja modeli** jest kluczowa - wszystkie modele przed kalibracją znacząco przeszacowywały prawdopodobieństwa

## 💻 Technologie

- **Język**: Python 3.7+
- **Analiza danych**: Pandas, NumPy
- **Modele ML**: Scikit-learn, XGBoost
- **Wizualizacja**: Matplotlib, Seaborn, Mplsoccer
- **Źródło danych**: StatsBombPy

## 📁 Struktura projektu
```
Football-xG-Predictor/
├── notebooks/                 
│   ├── data_collection.py      # Skrypt do zbierania danych
│   └── xg_model.ipynb          # Główny notebook z modelem xG
├── src/                        
│   ├── __init__.py             # Plik inicjalizujący pakiet
│   ├── preprocessing.py        # Funkcje do przetwarzania danych
│   ├── feature_engineering.py  # Inżynieria cech
│   ├── modeling.py             # Implementacja modeli
│   ├── evaluation.py           # Metryki i ocena modeli
│   └── visualization.py        # Wizualizacje
├── data/                       # Folder z danymi
├── assets/                     # Grafiki i wizualizacje
├── requirements.txt            # Zależności
└── README.md                   # Opis projektu/ ten plik
```

Uwaga: Projekt wykorzystuje publicznie dostępne dane StatsBomb, używane zgodnie z ich warunkami licencji.
