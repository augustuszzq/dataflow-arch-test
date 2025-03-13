import pandas as pd
import matplotlib.pyplot as plt
import re

# Load the uploaded CSV file
file_path = 'res_withmixed.csv'
df = pd.read_csv(file_path)

# Function to extract values using regex
def extract_value(text, pattern):
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None

# Initialize lists to store the extracted data
steps = []
losses = []
rates = []
mixed_precisions = []

# Extract data from each row
for _, row in df.iterrows():
    row_data = row[0]
    step = extract_value(row_data, r"Step=(\d+)")
    loss = extract_value(row_data, r"Loss=([\d.]+)")
    rate = extract_value(row_data, r"Rate=([\d.]+) samples/sec")
    mixed_precision = 'mixed_precision' in row_data

    if step is not None and loss is not None and rate is not None:
        steps.append(step)
        losses.append(loss)
        rates.append(rate)
        mixed_precisions.append(mixed_precision)

# Create a new DataFrame with the extracted data
data = {
    'step': steps,
    'loss': losses,
    'rate': rates,
    'mixed_precision': mixed_precisions
}
df_cleaned = pd.DataFrame(data)

# Separate data based on mixed precision
df_false = df_cleaned[df_cleaned['mixed_precision'] == False]
df_true = df_cleaned[df_cleaned['mixed_precision'] == True]

# Plotting loss
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(df_false['step'], df_false['loss'], label='Mixed Precision = False', marker='o')
plt.plot(df_true['step'], df_true['loss'], label='Mixed Precision = True', marker='o')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Loss vs. Steps')
plt.legend()

# Plotting rate
plt.subplot(1, 2, 2)
plt.plot(df_false['step'], df_false['rate'], label='Mixed Precision = False', marker='o')
plt.plot(df_true['step'], df_true['rate'], label='Mixed Precision = True', marker='o')
plt.xlabel('Steps')
plt.ylabel('Rate (samples/sec)')
plt.title('Rate vs. Steps')
plt.legend()

plt.tight_layout()
plt.show()
