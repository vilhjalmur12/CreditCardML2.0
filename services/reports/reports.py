import pandas as pd
import numpy as np

def parameterOverview(dataset, target_attribute):
    # Extracting # of unique entires per column and their sample values
    num_unique = []
    sample_col_values = []
    for col in dataset.columns:
        num_unique.append(len(dataset[col].unique()))  # Counting number of unique values per each column
        sample_col_values.append(dataset[col].unique()[:3])  # taking 3 sample values from each column

    # combining the sample values into a a=single string (commas-seperated)
    # ex)  from ['hi', 'hello', 'bye']  to   'hi, hello, bye'
    col_combined_entries = []
    for col_entries in sample_col_values:
        entry_string = ""
        for entry in col_entries:
            entry_string = entry_string + str(entry) + ', '
        col_combined_entries.append(entry_string[:-2])

    # Generating a list 'param_nature' that distinguishes features and targets
    param_nature = []
    for col in dataset.columns:
        if col == target_attribute:
            param_nature.append('Target')
        else:
            param_nature.append('Feature')

    # Generating Table1. Parameters Overview
    df_feature_overview = pd.DataFrame(np.transpose([param_nature, num_unique, col_combined_entries]),
                                       index=dataset.columns, columns=['Parameter Nature', '# of Unique Entries',
                                                                       'Sample Entries (First three values)'])
    return df_feature_overview