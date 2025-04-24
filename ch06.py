from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

file_path = os.environ.get("FILE_PATH")

df = pd.read_csv(
    file_path,
    index_col=["TimeGenerated"],
    usecols=["TimeGenerated", "TotalBytesSent"]
)

from msticpy.analysis.timeseries import timeseries_anomalies_stl

output = timeseries_anomalies_stl(df, seasonal=7)

#print(output[output.anomalies == 1])

import matplotlib.pyplot as plt

output["TimeGenerated"] = pd.to_datetime(output.index)
output = output.sort_values(by="TimeGenerated")

if "anomalies" not in output.columns:
    raise ValueError("The 'anomalies' column is missing in the output DataFrame.")

print(output.head(10))

# Plot the time series data
plt.figure(figsize=(12, 6))
plt.plot(output["TimeGenerated"], output["TotalBytesSent"], label="TotalBytesSent", color="blue")

# Highlight anomalies
anomalies = output[output["anomalies"] == 1]
plt.scatter(anomalies["TimeGenerated"], anomalies["TotalBytesSent"], color="red", label="Anomalies")

plt.xlabel("TimeGenerated")
plt.ylabel("TotalBytesSent")
plt.title("Time Series with Anomalies")
plt.legend()
plt.grid()
plt.show()

# Save the plot
output_path = os.environ.get("OUTPUT_PATH")
plt.savefig(output_path, format="png", dpi=300)
plt.close()
