import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Manually entered data points
thresholds = [0.00, 0.05, 0.1, 0.15, 0.2, 0.25]
asr_values = [0.6, 0.580, 0.45, 0.30, 0.1, 0.01]

# Create DataFrame for plotting
df = pd.DataFrame({
    'Threshold': thresholds,
    'ASR': asr_values
})

# Create plot using Seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Threshold', y='ASR', marker='o')

# Customize plot
plt.title('Attack Success Rate vs Threshold')
plt.xlabel('Fraction of data used for audit training (Threshold)')
plt.ylabel('Attack Success Rate (ASR)')
plt.grid(True)

# Save and show plot
plt.savefig('asr_vs_threshold.png')
plt.show()
