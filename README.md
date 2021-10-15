# MSc_Project
Ensemble Methods

## s_mv.py (Simple Majority Vote) & s_app.py (Simple Average Predicted Probability)

1. The program will automatically grab any .csv files in the directory so only all necessary files for a given run should be in there
    e.g., ABCpred outputs for EBV training,
          Bepipred2 outputs for EBV training,
          iBCE-EL outputs for EBV training,
          LBtope outputs for EBV training
2. Ground truth file needs to be in a separate folder and it's path entered manually to into line 10.
3. Enter path to where the final prediction** will be saved (line 76 in s_mv.py and line 71 in s_app.py)

## w_mv.py (Weighted Majority Vote) & w_app.py (Weighted Average Predicted Probability)

1. All file paths to be entered manually into their respective variables
2. Weights to be set accordingly (line 32) (see weights below)
3. Enter path to where the final prediction** will be saved (line 90)

## WEIGHTS

- EBV - Training        [0.50,	0.51,	0.49,	0.54]
- EBV - Holdout         [0.50,	0.51,	0.51,	0.54]

- HepC - Training       [0.56,	0.56,	0.44,	0.71]
- HepC - Holdout        [0.37,	0.61,	0.63,	0.73]

- Ovolvulus - Training  [0.41,	0.55,	0.59,	0.55]
- Ovolvulus - Holdout   [0.45,	0.54,	0.55,	0.55]

- Spyogenes - Training  [0.60,	0.53,	0.41,	0.53]
- Spyogenes - Holdout   [0.50,	0.50,	0.50,	0.80]

## stacking.py (Stacking)

1. All file paths entered manually into their respective variables (Lines 9-25; Lines 66-78)
2. Comment/uncomment variables to accommodate your desired prediction (Lines 9-25; Lines 66-78)
3. Enter path to where the final prediction** will be saved (line 115)

** All final predictions are saved into a table containing the columns:
    [Info_protein_id],
    [Info_pos] 
 necessary for the gather_results function in R



