import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings

from classificazione_explanation.explanation import explain
from classificazione_grezza.batch_classify_spuro import batch_classify, display_dict_models

warnings.filterwarnings("ignore", category=FutureWarning)




dataset = pd.read_csv("data.csv", sep=',')
del dataset['name']

corr = dataset.corr()


# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(10, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
square=True, linewidths=.2)
plt.show()



#seaborn.heatmap(corr, annot=True)
#plt.show()


# Seleziona la colonna target e effettua la classificazione
y = dataset['status']
X = dataset.drop('status', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
dict_models = batch_classify(X_train, y_train, X_test, y_test, no_classifiers=10, verbose=False)
pd.set_option('display.max_columns', 7)
display_dict_models(dict_models).to_csv('result_class_spura.cvs')

model = (dict_models.get("XGBClassifier"))['model']
