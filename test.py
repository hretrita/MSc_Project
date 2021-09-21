import pandas as pd
import numpy as np

d = {'Info_protein_ID': ['Protein 1', 'Protein 1', 'Protein 1', 'Protein 1', 'Protein 1', 'Protein 1', 'Protein 1', 'Protein 1', 'Protein 1', 'Protein 1', 'Protein 2', 'Protein 2'],
     'Info_pos_AA': [22, 23, 24, 25, 26, 34, 35, 36, 37, 38, 45, 46],
     0: [0.4, 0.3, 0.7, 0.6, 0.4, 0.2, 0.6, 0.7, 0.8, 0.7, 0.7, 0.2],
     1: [0.5, 0.7, 0.8, 0.9, 0.9, 0.1, 0.6, 0.5, 0.7, 0.5, 0.4, 0.4],
     'Class': [1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1]
     }
df = pd.DataFrame(data=d)


df = (df
      # Sort the values so everything is in order of protein and position
      .sort_values(['Info_protein_ID', 'Info_pos_AA'], ascending=True)

      # reset the index (just to keep things clean)
      .reset_index(drop=True)
      )

# Populate a new column called "section"
df['section'] = None

# Populate the necessaryvariables
last_protein = 'no_protein'
last_position = -1
prev_gt = 'no_class'
section = 1

# Iterate over each row using iterrows
for row in df.iterrows():
    protein = row[1]['Info_protein_ID']
    position = row[1]['Info_pos_AA']
    gt = row[1]['Class']

    # If the current Class matches the different class
    if gt == prev_gt or prev_gt == 'no_class':

        # If the current protein and last protein match, update the section label of the dataframe
        if protein == last_protein or last_protein == 'no_protein':

            # Check if the current position is a continuation of the last position, if so insert the section label as-is
            if position == last_position + 1 or last_position == -1:
                df.loc[row[0], 'section'] = f'{protein}:{section}'

            # If not, Increase the section variable, and then insert the section label
            else:
                section += 1
                df.loc[row[0], 'section'] = f'{protein}:{section}'

        # If the current protein is a new protein, restart the section naming
        else:
            section = 1
            df.loc[row[0], 'section'] = f'{protein}:{section}'

    #
    else:
        if protein == last_protein or last_protein == 'no_protein':
            section += 1
            df.loc[row[0], 'section'] = f'{protein}:{section}'

    # After evaluating the current protein, reassign the "last_" variables
    last_protein = protein
    last_position = position
    prev_gt = gt

# The groupby you want to do
desired_output = (
    df[['section', 0, 1]]
        .groupby('section')
        .mean()
        .reset_index()
)

print(df)
av = (df.groupby('section')[[0, 1]].mean()) # compute av prob
print(av)