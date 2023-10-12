
import shap
import matplotlib

matplotlib.use('TkAgg')


def explain(model, X_test):

    shap.initjs()
    explainer = shap.Explainer(model, seed=25)

    shap_values = explainer(X_test)
    print(shap_values.shape)
    # Visualizza e salva il grafico come immagine
    shap.plots.bar(shap_values, max_display=15)

    shap.summary_plot(shap_values, X_test, plot_type='violin')  #Poiché il modello XGBoost ha una perdita logistica, l'asse x ha unità di log-odds

    # Visualizza e salva il grafico come immagine
    shap.plots.waterfall(shap_values[0])

    # Visualizza e salva il grafico come immagine
    shap_values = explainer(X_test.round(3))
    shap.plots.force(shap_values[0], matplotlib=True)


