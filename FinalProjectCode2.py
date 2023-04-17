# IMPORT STATEMENTS
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# GET REQUEST & HTML CONTENT
url = 'https://www.cryptoslam.io/nftglobal'
response = requests.get(url)

# PARSE HTML CONTENT BY BS4
soup = BeautifulSoup(response.content, 'html.parser')

# LOCATE TABLE
table = soup.find('table')

# CONVERT TABLE ROWS & COLUMNS TO DICT
table_rows = table.find_all('tr')
data = []
for row in table_rows:
    cells = row.find_all('td')
    if len(cells) > 0:
        item = {
            'Name': cells[0].text.strip(),
            'Price': float(cells[1].text.strip().replace('$', '').replace(',', '')),
            'Sales Volume': int(cells[2].text.strip().replace(',', '')),
            'Sales Value': float(cells[3].text.strip().replace('$', '').replace(',', '')),
            'Owners': int(cells[4].text.strip().replace(',', '')),
            'Buyers': int(cells[5].text.strip().replace(',', ''))
        }
        data.append(item)

# PANDAS DATAFRAME
df = pd.DataFrame(data)

# LIN REG OF MULT VARS
X = df[['Sales Volume', 'Owners', 'Buyers']]
Y = df['Price']
model = LinearRegression()
model.fit(X, Y)
coef = model.coef_
intercept = model.intercept_
print('Coefficients:', coef)
print('Intercept:', intercept)

# Create a pandas DataFrame from the list of dictionaries
df = pd.DataFrame(data)

# Calculate the linear regression for multiple variables
# In Progress
X = df[['Sales Volume', 'Owners', 'Buyers']]
Y = df['Price']
model = LinearRegression()
model.fit(X, Y)
# TO BE CONTINUED
