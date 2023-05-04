# IMPORT STATEMENTS
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# SCRAPE DATA
url = "https://www.cryptoslam.io/nftglobal"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

# EXTRACT DATA FROM HTML TABLE
table = soup.find("table")
rows = table.find_all("tr")
# STORE IN PANDAS DATAFRAME
data = []
for row in rows[1:]:
    cols = row.find_all("td")
    cols = [col.text.strip() for col in cols]
    data.append(cols)
columns = ["rank", "name", "owner", "volume_7d", "sales_volume", "number_of_owners", "number_of_buyers", "price"]
df = pd.DataFrame(data, columns=columns)

# CONVERT DATA TYPES
df["rank"] = df["rank"].astype(int)
df["volume_7d"] = df["volume_7d"].astype(float)
df["sales_volume"] = df["sales_volume"].astype(float)
df["number_of_owners"] = df["number_of_owners"].astype(int)
df["number_of_buyers"] = df["number_of_buyers"].astype(int)
df["price"] = df["price"].str.replace(",", "").astype(float)

# PERFORM LINEAR REGRESSION
X = df[["sales_volume", "number_of_owners", "number_of_buyers"]]
y = df["price"]
reg = LinearRegression().fit(X, y)

# PRINT MODEL COEFFICIENTS
print("Coefficients: ", reg.coef_)
print("Intercept: ", reg.intercept_)

# MAKE PREDICTIONS
new_data = np.array([[100, 10, 20], [200, 20, 30]])
predictions = reg.predict(new_data)
print("Predictions: ", predictions)

# VISUALIZE DATA
fig, ax = plt.subplots()
ax.scatter(df["sales_volume"], df["price"])
ax.plot(df["sales_volume"], reg.predict(X), color="red")
ax.set_xlabel("Sales Volume")
ax.set_ylabel("Price")
ax.set_title("Linear Regression Model")
plt.show()
