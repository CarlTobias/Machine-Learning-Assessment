import requests
import pandas as pd
import gradio as gr

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Importing from WhatToMine for Mining Data and fetching data in a JSON format
jsonResponse = 'https://whattomine.com/coins.json'
jsonWTM = requests.get(jsonResponse).json()

# Putting the formatted data into a workable DataFrame
WTMData = pd.DataFrame.from_dict(jsonWTM['coins'], orient = 'index')

print(f'{WTMData.info()}\n\n')

# Feature Scaling in order to amplify the accuracy of the model
WTMData['reward_efficiency'] = WTMData['block_reward'] / WTMData['difficulty']
WTMData['hashrate_efficiency'] = WTMData['nethash'] / WTMData['difficulty']
WTMData['normalized_profitability'] = WTMData['profitability'] / WTMData['exchange_rate']
WTMData['difficulty_change'] = WTMData['difficulty24'] / WTMData['difficulty']
WTMData['profitability_interaction'] = (WTMData['exchange_rate'] * WTMData['block_reward'] / WTMData['difficulty'])

WTMData['timestamp'] = pd.to_datetime(WTMData['timestamp'], unit = 's')

WTMData = WTMData.sort_values(by = 'timestamp')
WTMData['profitability_weekly_avg'] = WTMData['profitability'].rolling(window = 7).mean()
WTMData['exchange_rate_volatility'] = WTMData['exchange_rate'].rolling(window = 7).std()


# A list of all features (natural and derived)
numericColumns = [
    'block_reward', 'difficulty', 'nethash', 'exchange_rate', 
    'exchange_rate_vol', 'profitability', 'reward_efficiency', 
    'hashrate_efficiency', 'normalized_profitability', 'difficulty_change', 
    'profitability_interaction', 'profitability_weekly_avg', 'exchange_rate_volatility'
]

# Calling all features from the dataframe
numericData = WTMData[numericColumns]

# Using .corr() to observe the correlation matrix between features
correlationMatrix = numericData.corr()
print(f'Correlation: {correlationMatrix}')

# Plotted version of the printed correlation matrix
#plt.figure(figsize = (10, 8))
#sns.heatmap(correlationMatrix, annot = True, cmap = 'coolwarm', fmt = '.2f')
#plt.title('Correlation Matrix')
#plt.show()


# Splitting data
modelX = [
    'block_reward', 'difficulty', 'nethash', 
    'exchange_rate', 'exchange_rate_vol', 
    'reward_efficiency', 'hashrate_efficiency', 
    'normalized_profitability', 'difficulty_change', 
    'profitability_interaction', 'profitability_weekly_avg', 
    'exchange_rate_volatility'
]

modelY = 'profitability'

x = WTMData[modelX]
y = WTMData[modelY]


# Training the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)



# Running the Model
rfrModel = RandomForestRegressor(n_estimators = 100, random_state = 42)
rfrModel.fit(x_train, y_train)

yPredTrain = rfrModel.predict(x_train)
yPredTest = rfrModel.predict(x_test)

# Checking the accuracy of the model
mae = mean_absolute_error(y_test, yPredTest)
mse = mean_squared_error(y_test, yPredTest)
rmse = mse ** 0.5
r2 = r2_score(y_test, yPredTest)

print(f'\n\nMAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')



# GUI Interface with Gradio

# The for loop here takes the 'tag' data from jsonWTM
WTMDataTags = []
for coin in jsonWTM['coins'].values():
    WTMDataTags.append(coin['tag'])
 
WTMData['tag'] = WTMDataTags
WTMData.set_index('tag', inplace = True)

# The for loop here takes those tag names and puts them in a list
coinNames = []
for coins in jsonWTM['coins'].values():
    coinNames.append(coins['tag'])


# The user-defined function here takes the selected coin, runs it through the database, and if the selected coins is available, the program will predict the profitability of said coin
def selectedCrypto(selected):
    print(selected)
    if selected in WTMData.index:
        # The use of a try-except loop isn't necessary, but I used it as a debugging tool to find out where errors were happening during the coding process
        try:
            coinsData = WTMData.loc[selected, modelX].values.reshape(1, -1)
            print(f"coinData: {coinsData}")
            
            profitabilityPred = rfrModel.predict(coinsData)[0]
            
            difficulty = coinsData[0][modelX.index('difficulty')]
            nethash = coinsData[0][modelX.index('nethash')]
            profitability = coinsData[0][modelY.index('profitability')]
            
            statistics = f"""
            Selected Coin: {selected}
            Difficulty: {difficulty}
            Nethash: {nethash}
            Current Profitability: {profitability}
            Predicted Profitability: {profitabilityPred}
            """
            
            return statistics
        
        except Exception as x:
            print(f"Error during prediction: {x}")
            
            return "Error: Unable to calculate profitability"
    
    else:
        return "Error: Selected cryptocurrency not available"
    
def compareCrypto(coinOne, coinTwo):
    if coinOne in WTMData.index and coinTwo in WTMData.index:
        try:
            coinOneData = WTMData.loc[coinOne, modelX].values.reshape(1, -1)
            coinTwoData = WTMData.loc[coinTwo, modelX].values.reshape(1, -1)
            
            profPredOne = rfrModel.predict(coinOneData)[0]
            profPredTwo = rfrModel.predict(coinTwoData)[0]
            
            
            comparison = f"""
            {coinOne}'s Predicted Profitability: {profPredOne}
            {coinTwo}'s Predicted Profitability: {profPredTwo}
            """
            
            coinOneStatistics = f"""
            Selected Coin: {coinOne}
            Difficulty: {coinOneData[0][modelX.index('difficulty')]}
            Nethash: {coinOneData[0][modelX.index('nethash')]}
            Current Profitability: {coinOneData[0][modelY.index('profitability')]}
            """
            
            coinTwoStatistics = f"""
            Selected Coin: {coinTwo}
            Difficulty: {coinTwoData[0][modelX.index('difficulty')]}
            Nethash: {coinTwoData[0][modelX.index('nethash')]}
            Current Profitability: {coinTwoData[0][modelY.index('profitability')]}
            """
            
            return f"{comparison}\n\n{coinOneStatistics}\n\n{coinTwoStatistics}"
        
        except Exception as x:
            print(f"Error during prediction: {x}")
            
            return "Error: Unable to calculate profitability"
    
    else:
        return "Error: Selected cryptocurrency not available"





# Here is where the GUI is located
with gr.Blocks() as interface:
    # Markdown sets the title text
    gr.Markdown("Cryptocurrency Mining Profitability")
    # I used a dropdown menu for easy selection
    dropdownMenu = gr.Dropdown(label = "Select your crypto", choices = coinNames)
    
    # Output is dislpayed below the dropdown menu in a textbox
    output = gr.Textbox(label = "Predicted Mining Profitability")
    
    # Added a button so the user has the option to run the program whenever
    predictButton = gr.Button("Calculate")
    predictButton.click(selectedCrypto, inputs = dropdownMenu, outputs = output)
    
    
    # As an extra, show a graph of the selected coin with the profitability or the correlation of the other features that predict
    cOneDropdown = gr.Dropdown(label = "Select First Coin", choices = coinNames)
    cTwoDropdown = gr.Dropdown(label = "Select Second Coin", choices = coinNames)
    
    cOutput = gr.Textbox(label = "Comparison of Cryptocurrencies")
    
    cButton = gr.Button("Compare")
    cButton.click(compareCrypto, inputs = [cOneDropdown, cTwoDropdown], outputs = cOutput)
    

# Launching the interface
interface.launch(share = True)