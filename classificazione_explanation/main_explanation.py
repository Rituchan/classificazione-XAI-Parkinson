import shap
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
from classificazione_explanation.batch_classify_explenation import batch_classify, display_dict_models
from classificazione_explanation.explanation import explain
from classificazione_explanation.normalize import normalize_dataset

warnings.filterwarnings("ignore", category=FutureWarning)

df = pd.read_csv("data.csv", sep=',')
del df['name']

missing_count = df.isnull().sum()

filtered_data = df[
    (df['MDVP:Flo(Hz)'] >= 70) &
    (df['MDVP:Flo(Hz)'] <= 350) &
    (df['MDVP:Fhi(Hz)'] >= 70) &
    (df['MDVP:Fhi(Hz)'] <= 350)
    ]

# Stampa il numero di righe nel dataset originale e nel dataset filtrato
print("\nNumero di righe nel dataset originale:", len(df))
print("Numero di righe nel dataset filtrato:", len(filtered_data))

# Esporto il dataset filtrato in un nuovo file CSV:
filtered_data.to_csv("data_filtered.csv", index=False)

filtered_data = pd.read_csv("data_filtered.csv", sep=',')

valori_Fhi = filtered_data['MDVP:Fhi(Hz)']
media_Fhi = valori_Fhi.mean()
deviazione_standard_Fhi = valori_Fhi.std()

valori_spread1 = filtered_data['spread1']
media_spread1 = valori_spread1.mean()
deviazione_standard_spread1 = valori_spread1.std()

valori_spread2 = filtered_data['spread2']
media_spread2 = valori_spread2.mean()
deviazione_standard_spread2 = valori_spread2.std()

valori_RAP = filtered_data['MDVP:RAP']
media_RAP = valori_RAP.mean()
deviazione_standard_RAP = valori_RAP.std()

print("\n\n")
print("Media Fhi:", media_Fhi)
print("Deviazione standard Fhi:", deviazione_standard_Fhi)
print("\n\n")

print("Media spread1:", media_spread1)
print("Deviazione standard spread 1:", deviazione_standard_spread1)
print("\n\n")

print("Media spread2:", media_spread2)
print("Deviazione standard spread2:", deviazione_standard_spread2)
print("\n\n")

print("Media RAP:", media_RAP)
print("Deviazione standard RAP:", deviazione_standard_RAP)
print("\n\n")


pd.set_option('display.max_rows', 1000)
sourceFile = open('description_data_type.txt', 'w')
print(filtered_data.dtypes, file=sourceFile)
sourceFile.close()

##normalizziamo il dataset
(min_value_Fhi, max_value_Fhi, min_spread1, max_spread1, min_spread2, max_spread2, min_RAP, max_RAP) = normalize_dataset(filtered_data, 'normalized_data.csv')
print("\n\n", "Min value MDPV(Fhi) = ", min_value_Fhi, "\n", "Max value MDPV(Fhi) = ", max_value_Fhi, "\n\n")
print("\n\n", "Min value spread1 = ", min_spread1, "\n", "Max value spread1 = ", max_spread1, "\n\n")
print("\n\n", "Min value spread2 = ", min_spread2, "\n", "Max value spread2 = ", max_spread2, "\n\n")
print("\n\n", "Min value RAP = ", min_RAP, "\n", "Max value RAP = ", max_RAP, "\n\n")

df_normalize = pd.read_csv("normalized_data.csv", sep=',')
df_normalize["status"] = df_normalize["status"].astype(int)

# Applica la tecnica di oversampling SMOTE per bilanciare la classe target
y = df_normalize['status']
X = df_normalize.drop('status', axis=1)
smote = SMOTE(sampling_strategy='not majority', random_state=1)
X, y = smote.fit_resample(X, y)
df_res = pd.concat([X, y], axis=1)
df_res.to_csv('dataset_bilanciato.csv', index=False)

# carichiamo il dataset per la classficazione
df_classificazione = pd.read_csv("dataset_bilanciato.csv", sep=',')

# Seleziona la colonna target ed effettua la classificazione
y = df_classificazione['status']
X = df_classificazione.drop('status', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
dict_models, y_pred = batch_classify(X_train, y_train, X_test, y_test, no_classifiers=10)
pd.set_option('display.max_columns', 7)
display_dict_models(dict_models).to_csv('data_result.cvs')

clustering = shap.utils.hclust(
    X_test, y_pred
)

model = (dict_models.get("XGBClassifier"))['model']
explain(model, X_test)
