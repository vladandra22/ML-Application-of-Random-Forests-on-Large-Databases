import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import math

# Informatii despre date.
def analiza_atribute_numerice_continue(df, numeric_columns):
    print("----- ANALIZA ATRIBUTE NUMERICE CONTINUE -----")
    # .T = transpose, transpunem matricea pentru a vedea mai frumos datele
    return df[numeric_columns].describe().T

# Vizualizare grafice atribute numerice continue
def boxplots_atribute_numerice_continue(df, numeric_columns):
    plt.figure(figsize=(10, 5))
    for idx in range(0, len(numeric_columns)):
        cols = math.ceil(math.sqrt(len(numeric_columns))) 
        rows = math.ceil(len(numeric_columns) / cols)  
        plt.subplot(cols, rows, idx + 1)
        sns.boxplot(x=df[numeric_columns[idx]])
        plt.title(f'Boxplot pentru atributul {numeric_columns[idx]}')
    plt.tight_layout()
    plt.show()

def analiza_atribute_discrete(df, categorical_columns):
    print("----- ANALIZA ATRIBUTE DISCRETE SAU ORDINALE -----")
    stats = pd.DataFrame(index=categorical_columns, columns=['count', 'unique'])
    # Exemple care nu au valori lipsa
    stats['count'] = df[categorical_columns].notnull().sum()
    # Valori unice
    stats['unique'] = df[categorical_columns].nunique()
    return stats

# Vizualizare histograme atribute discrete sau ordinale
def histograms_atribute(df, categorical_columns):
    # Facem mai multe ploturi daca sunt foarte multe coloane si nu incap intr-unul singur.
    for col in categorical_columns:
        plt.figure(figsize=(20, 5))
        sns.countplot(x=df[col])
        plt.title(f'Distributia pentru {col}')
        plt.show()


# Count plot pentru a reprezenta frecventa de aparitie a fiecarei etichete
# 'Diabet' pentru Diabet, 'loan_approval_status' pt credit
def plot_frecventa_clase(df, nume, tip):
    plt.figure(figsize=(10, 5))
    df[nume].hist(bins=len(df[nume].unique()))
    plt.title(f'Frecventa clase in setul de date de {tip}')
    plt.show()


def matrice_de_corelatie(df, numeric_columns, categorical_columns):
    # Foloseste by default Pearson
    pearson_mat = df[numeric_columns].corr()

    # Avem redundante intre doua atribute numerice daca
    # avem o valoare mai mare de 0.75 in matricea Pearson.
    redundant_numeric= []
    for i in range(len(pearson_mat)):
        for j in range(i + 1, len(pearson_mat)):
            if abs(pearson_mat.iloc[i, j]) >= 0.75:
                redundant_numeric.append((numeric_columns[i], numeric_columns[j]))

    
    redundant_categorical = []
    mat_size = len(categorical_columns)
    chi2_mat = np.zeros((mat_size, mat_size))
    for i, col1 in enumerate(categorical_columns):
        for j, col2 in enumerate(categorical_columns):
            if i < j:
                table = pd.crosstab(df[col1], df[col2])
                chi2, p, _, _ = chi2_contingency(table)
                if p < 0.001:
                        redundant_categorical.append((p, col1, col2))
                chi2_mat[i, j] = chi2
                chi2_mat[j, i] = chi2
            elif i == j:
                chi2_mat[i, j] = 1
    redundant_categorical.sort()
    return (pearson_mat, redundant_numeric, chi2_mat, redundant_categorical)
    
def plot_matrice_de_corelatie(pearson_mat, numeric_columns, chi2_mat, categorical_columns):
    # Foloseste by default Pearson
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    # Normalizam datele folosind vmin, vmax
    cax = ax.matshow(pearson_mat, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, 6, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(numeric_columns)
    ax.set_yticklabels(numeric_columns)
    # Adaugam si etichete cu valoarea.
    for i in range(len(numeric_columns)):
        for j in range(len(numeric_columns)):
            ax.text(j, i, round(pearson_mat.iloc[i, j], 2), ha='center', va='center', color='w')
    plt.title('Matricea de Corelație pentru Atribute Numerice', pad=20)
    plt.show()
  
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(chi2_mat, cmap='coolwarm') 
    fig.colorbar(cax)
    ticks = np.arange(0, len(categorical_columns), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(categorical_columns)
    ax.set_yticklabels(categorical_columns)
    plt.title('Matricea Chi-pătrat pentru Atribute Categorice', pad=20)
    plt.show()


# Valorile extreme sunt tratate ca si valori lipsa. 
def eliminare_valori_extreme(df, numerical_columns):
    for col in numerical_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        # https://www.analyticsvidhya.com/blog/2022/09/dealing-with-outliers-using-the-iqr-method/
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].mask((df[col] < lower_bound) | (df[col] > upper_bound))
    return df

# Preprocess data
def preprocesare_data(df, categorical_columns, numeric_columns):
    # 1) Valorile extreme vor fi eliminate (tratate ca si valori lipsa)
    # Aceasta este folositoare pentru Regresia Logistica, insa am observat
    # ca obtinem rezultate mai bune pentru paduri aleatoare, asa ca o lasam comentata.
    # df = eliminare_valori_extreme(df, numeric_columns)
    # 2) Imputam date pentru valorile lipsa.
    imputer_categorical = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = imputer_categorical.fit_transform(df[categorical_columns])
    # Foloseste o metoda iterativa pentru a estima valorile lipsa. Folosim acelasi random state
    # de la care sa plecam mereu (0) pentru a pastra codul determinist. 
    imputer_numerical = IterativeImputer(random_state = 0)
    df[numeric_columns] = imputer_numerical.fit_transform(df[numeric_columns])

    # 3) Standardizam datele.
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # 4) Eliminam atributele redundante.
    (_, redundante_numeric, _, redundante_categorical) = matrice_de_corelatie(df, numeric_columns, categorical_columns)
    
    # Luam al doilea atribut din redundantele numerice.
    atr_redundante_numeric = [b for (a, b) in redundante_numeric if b in df.columns]
    # Luam al doilea atribut din redundantele categorice, in afara de 'Diabetes'.
    # Eliminam primele 5 atribute redundante, cu p-value cel mai mic.
    atr_redundante_categoric = [b for (p, a, b) in redundante_categorical[:5] if b != 'Diabetes' and b in df.columns]

    # Dam drop atributelor redundante.
    df = df.drop(columns=atr_redundante_numeric + atr_redundante_categoric)
    return df

def train_si_eval_model_random_forest(train_data, test_data, full_data, numeric_columns, categorical_columns):
    train_data = preprocesare_data(train_data, categorical_columns, numeric_columns)
    test_data = preprocesare_data(test_data, categorical_columns, numeric_columns)
    full_data = preprocesare_data(full_data, categorical_columns, numeric_columns)

    X_train = train_data.drop(columns=['Diabetes'])
    Y_train = train_data['Diabetes']

    X_test = test_data.drop(columns=['Diabetes'])
    Y_test = test_data['Diabetes']

    Y_test = Y_test.astype('int')
    Y_train = Y_train.astype('int')

    print("\nDataset: Diabetes")
    print("Train set size: ", len(X_train))
    print("Test set size: ", len(X_test))
    print("Number of classes: ", Y_train.nunique())
    print("Number of features: ", len(X_train.columns))
    print("\nNumber of examples per class in train set:")
    print(Y_train.value_counts())

    # Convertim variabilele categorice in variabile numerice folosind One-Hot Encoding
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    X_train = X_train.reindex(columns=X_test.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Antrenam RandomForestClassifier
    model = RandomForestClassifier(n_estimators=50, random_state=3)
    model.fit(X_train, Y_train)
    # Facem predictii pe setul de test
    y_pred = model.predict(X_test)
    # Evaluam modelul
    accuracy = accuracy_score(Y_test, y_pred)
    report = classification_report(Y_test, y_pred,  zero_division=0)

    print(f"\nAccuracy: {accuracy}")
    print(f"Classification Report: \n{report}")

def solve():
    full_filepath = 'Diabet_full.csv'
    train_filepath = 'Diabet_train.csv'
    test_filepath = 'Diabet_test.csv'

    df = pd.read_csv(full_filepath)
    df_train = pd.read_csv(train_filepath)
    df_test = pd.read_csv(test_filepath)

    # Definim atributele numerice si atributele categorice.
    numeric_columns = ['psychological-rating', 'BodyMassIndex', 'Age', 'CognitionScore', 'Body_Stats', 'Metabolical_Rate']
    categorical_columns = [
        'HealthcareInterest', 'PreCVA', 'RoutineChecks', 'CompletedEduLvl', 'alcoholAbuse',
        'cholesterol_ver', 'vegetables', 'HighBP', 'Unprocessed_fructose', 'Jogging', 'IncreasedChol',
        'gender', 'HealthScore', 'myocardial_infarction', 'SalaryBraket', 'Cardio', 'ImprovedAveragePulmonaryCapacity',
        'Smoker', 'Diabetes'
    ]

    stats_atribute_numerice = analiza_atribute_numerice_continue(df, numeric_columns)
    print(stats_atribute_numerice)
    boxplots_atribute_numerice_continue(df, numeric_columns)
    stats_atribute_discrete = analiza_atribute_discrete(df, categorical_columns)
    print(stats_atribute_discrete)
    histograms_atribute(df, categorical_columns)
    plot_frecventa_clase(df_train, 'Diabetes', 'antrenare')
    plot_frecventa_clase(df_test, 'Diabetes', 'testare')
    # Pentru analizia corelatiei intre atribute, voi realiza matricea
    # de corelatie pentru ambele tipuri de atribute.
    (pearson_mat, redundante_numerical, chi2_mat, redundante_categorical) = matrice_de_corelatie(df, numeric_columns, categorical_columns)
    plot_matrice_de_corelatie(pearson_mat, numeric_columns, chi2_mat, categorical_columns)
    train_si_eval_model_random_forest(df_train, df_test, df, numeric_columns, categorical_columns)
