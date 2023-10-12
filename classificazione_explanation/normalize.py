def normalize_dataset(input_dataset, output_file):
    # Copia il dataset originale per non modificarlo direttamente

    global min_value_Fhi, max_value_Fhi, min_spread1, max_spread1, min_spread2, max_spread2, min_RAP, max_RAP
    normalized_data = input_dataset.copy()

    # Itera su ciascuna colonna numerica per normalizzare tra 0 e 1
    for feature_name in input_dataset.columns:
        if input_dataset[feature_name].dtype == 'float64':
            max_value = input_dataset[feature_name].max()
            min_value = input_dataset[feature_name].min()
            normalized_data[feature_name] = (input_dataset[feature_name] - min_value) / (max_value - min_value)
            if "MDVP:Fhi(Hz)" in feature_name:
                min_value_Fhi = min_value
                max_value_Fhi = max_value

            if "spread1" in feature_name:
                min_spread1 = min_value
                max_spread1 = max_value

            if "spread2" in feature_name:
                min_spread2 = min_value
                max_spread2 = max_value

            if "spread2" in feature_name:
                min_spread2 = min_value
                max_spread2 = max_value

            if "MDVP:RAP" in feature_name:
                min_RAP = min_value
                max_RAP = max_value

    # Salva il DataFrame normalizzato in un nuovo file CSV
    normalized_data.to_csv(output_file, index=False)

    return min_value_Fhi, max_value_Fhi, min_spread1, max_spread1, min_spread2, max_spread2, min_RAP, max_RAP
