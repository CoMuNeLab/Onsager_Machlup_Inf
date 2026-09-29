import pandas as pd

# Define the genes in the order they appear in the file
genes = ['CBF1', 'GAL4', 'SWI5', 'GAL80', 'ASH1']

# Parse Switch-On
df_on = pd.read_excel('1-s2.0-S0092867409001561-mmc2.xls', sheet_name=' Average of all Switch-on Data', header=None)
df_on = df_on.iloc[1:, :6] # Skip header row, take first 6 columns (Time + 5 genes)
df_on.columns = ['Time_min'] + genes
# Drop any completely empty rows
df_on = df_on.dropna(how='all')
df_on.to_csv('irma_switch_on.csv', index=False)

# Parse Switch-Off
df_off = pd.read_excel('1-s2.0-S0092867409001561-mmc2.xls', sheet_name='Average of all Switch-off data', header=None)
df_off = df_off.iloc[1:, :6]
df_off.columns = ['Time_min'] + genes
df_off = df_off.dropna(how='all')
df_off.to_csv('irma_switch_off.csv', index=False)

print("Successfully extracted irma_switch_on.csv and irma_switch_off.csv!")
