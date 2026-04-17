# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd

# ==============================
# 2. Load Dataset
# ==============================
df = pd.read_csv(r"C:\Users\adity\OneDrive\Documents\reddit_depression_dataset.csv")

print("Original Columns:", df.columns)

# ==============================
# 3. Fix Column Name
# ==============================
df.rename(columns={'lable': 'label'}, inplace=True)

# ==============================
# 4. Combine Title + Body
# ==============================
df['text'] = df['title'].fillna('') + " " + df['body'].fillna('')

# ==============================
# 5. Keep Required Columns
# ==============================
df = df[['text', 'label']]

# ==============================
# 6. Remove Missing Values
# ==============================
df.dropna(inplace=True)

# ==============================
# 7. Check Labels
# ==============================
print("Unique Labels:", df['label'].unique())

# If labels are strings, convert them
# Uncomment if needed:

# df['label'] = df['label'].map({
#     'depression': 1,
#     'not_depression': 0
# })

# ==============================
# 8. Save Clean Dataset
# ==============================
df.to_csv('cleaned_dataset.csv', index=False)

print("Cleaned dataset saved!")