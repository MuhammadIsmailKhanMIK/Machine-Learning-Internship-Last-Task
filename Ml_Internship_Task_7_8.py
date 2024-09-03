# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load the dataset from a CSV file
data = pd.read_csv('COVID-19 Daily.csv')

# Print column names to verify the data structure
print("Column names:", data.columns)

# Remove any leading or trailing whitespace from column headers
data.columns = data.columns.str.strip()

# Convert 'Date' column to datetime format
# If conversion fails, 'errors='coerce'' will turn invalid parsing into NaT (Not a Time)
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Ensure all relevant columns are numeric
numeric_columns = ['Daily Tests', 'Daily Cases', 'Daily Recoveries', 'Daily Deaths']
for column in numeric_columns:
    data[column] = pd.to_numeric(data[column], errors='coerce')

# Drop rows where any of the relevant columns have NaN values
# This is necessary after conversion to datetime and numeric types to ensure data integrity
data.dropna(subset=['Date'] + numeric_columns, inplace=True)

# Print information about missing values, column types, and basic statistics
print("\nNull values in each column:")
print(data.isnull().sum())
print("\nColumn info:")
print(data.info())
print("\nDescriptive statistics:")
print(data.describe())

# Analysis and plotting for 'Daily Tests' if the column exists
if 'Daily Tests' in data.columns:
    # Calculate the total number of tests performed
    total_tests = data['Daily Tests'].sum()
    print(f'\nTotal tests performed till September 01, 2020: {total_tests}')

    # Plot Daily Testing Trend Line Graph
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['Daily Tests'], marker='o')
    plt.title('Daily Testing Trend Line Graph')
    plt.xlabel('Date')
    plt.ylabel('Daily Tests')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.show()

    # Plot Scattered Diagram of Daily Testing
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Date'], data['Daily Tests'])
    plt.title('Scattered Diagram of Daily Testing')
    plt.xlabel('Date')
    plt.ylabel('Daily Tests')
    plt.xticks(rotation=45)
    plt.show()

    # Plot Bar Graph of Daily Testing
    plt.figure(figsize=(10, 6))
    plt.bar(data['Date'], data['Daily Tests'])
    plt.title('Bar Graph of Daily Testing')
    plt.xlabel('Date')
    plt.ylabel('Daily Tests')
    plt.xticks(rotation=45)
    plt.show()

    # Scatter diagram of above-average (red) and below-average (green) daily testing
    avg_test = data['Daily Tests'].mean()
    above_avg = data[data['Daily Tests'] > avg_test]
    below_avg = data[data['Daily Tests'] <= avg_test]

    plt.figure(figsize=(10, 6))
    plt.scatter(above_avg['Date'], above_avg['Daily Tests'], color='red', label='Above Average')
    plt.scatter(below_avg['Date'], below_avg['Daily Tests'], color='green', label='Below Average')
    plt.title('Scatter Diagram of Above and Below Average Daily Testing')
    plt.xlabel('Date')
    plt.ylabel('Daily Tests')
    plt.legend()  # Add a legend to differentiate between above and below average
    plt.xticks(rotation=45)
    plt.show()

# Perform the same analysis for 'Daily Cases', 'Daily Recoveries', and 'Daily Deaths'
for column in ['Daily Cases', 'Daily Recoveries', 'Daily Deaths']:
    if column in data.columns:
        # Calculate the total number of each metric
        print(f'\nTotal {column} till September 01, 2020: {data[column].sum()}')
        
        # Trend Line Graph
        plt.figure(figsize=(10, 6))
        plt.plot(data['Date'], data[column], marker='o')
        plt.title(f'Daily {column} Trend Line Graph')
        plt.xlabel('Date')
        plt.ylabel(column)
        plt.xticks(rotation=45)
        plt.show()

        # Scattered Diagram
        plt.figure(figsize=(10, 6))
        plt.scatter(data['Date'], data[column])
        plt.title(f'Scattered Diagram of Daily {column}')
        plt.xlabel('Date')
        plt.ylabel(column)
        plt.xticks(rotation=45)
        plt.show()

        # Bar Graph
        plt.figure(figsize=(10, 6))
        plt.bar(data['Date'], data[column])
        plt.title(f'Bar Graph of Daily {column}')
        plt.xlabel('Date')
        plt.ylabel(column)
        plt.xticks(rotation=45)
        plt.show()

        # Scatter diagram of above-average (red) and below-average (green) values
        avg = data[column].mean()
        above_avg = data[data[column] > avg]
        below_avg = data[data[column] <= avg]

        plt.figure(figsize=(10, 6))
        plt.scatter(above_avg['Date'], above_avg[column], color='red', label='Above Average')
        plt.scatter(below_avg['Date'], below_avg[column], color='green', label='Below Average')
        plt.title(f'Scatter Diagram of Above and Below Average Daily {column}')
        plt.xlabel('Date')
        plt.ylabel(column)
        plt.legend()
        plt.xticks(rotation=45)
        plt.show()

# Convert the dataset into weekly and monthly statistics
data['Week'] = data['Date'].dt.to_period('W')  # Convert dates to weekly periods
data['Month'] = data['Date'].dt.to_period('M')  # Convert dates to monthly periods

# Aggregate data by week and month, summing up the relevant columns
weekly_data = data.groupby('Week')[numeric_columns].sum()
monthly_data = data.groupby('Month')[numeric_columns].sum()

# Perform the same analysis for weekly and monthly data
for column in numeric_columns:
    if column in data.columns:
        # Print total weekly and monthly values
        print(f'\nTotal weekly {column} till September 01, 2020: {weekly_data[column].sum()}')
        print(f'Total monthly {column} till September 01, 2020: {monthly_data[column].sum()}')
        
        # Weekly Analysis
        plt.figure(figsize=(10, 6))
        plt.plot(weekly_data.index.astype(str), weekly_data[column], marker='o')
        plt.title(f'Weekly {column} Trend Line Graph')
        plt.xlabel('Week')
        plt.ylabel(column)
        plt.xticks(rotation=45)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.scatter(weekly_data.index.astype(str), weekly_data[column])
        plt.title(f'Scattered Diagram of Weekly {column}')
        plt.xlabel('Week')
        plt.ylabel(column)
        plt.xticks(rotation=45)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.bar(weekly_data.index.astype(str), weekly_data[column])
        plt.title(f'Bar Graph of Weekly {column}')
        plt.xlabel('Week')
        plt.ylabel(column)
        plt.xticks(rotation=45)
        plt.show()

        # Monthly Analysis
        plt.figure(figsize=(10, 6))
        plt.plot(monthly_data.index.astype(str), monthly_data[column], marker='o')
        plt.title(f'Monthly {column} Trend Line Graph')
        plt.xlabel('Month')
        plt.ylabel(column)
        plt.xticks(rotation=45)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.scatter(monthly_data.index.astype(str), monthly_data[column])
        plt.title(f'Scattered Diagram of Monthly {column}')
        plt.xlabel('Month')
        plt.ylabel(column)
        plt.xticks(rotation=45)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.bar(monthly_data.index.astype(str), monthly_data[column])
        plt.title(f'Bar Graph of Monthly {column}')
        plt.xlabel('Month')
        plt.ylabel(column)
        plt.xticks(rotation=45)
        plt.show()

# Find maximum and minimum number of weekly & monthly tests, cases, recoveries, and deaths along with their dates
for column in numeric_columns:
    if column in data.columns:
        # Find max and min values and their corresponding dates for weekly data
        max_weekly = weekly_data[column].max()
        min_weekly = weekly_data[column].min()
        max_weekly_date = weekly_data[column].idxmax()
        min_weekly_date = weekly_data[column].idxmin()

        # Find max and min values and their corresponding dates for monthly data
        max_monthly = monthly_data[column].max()
        min_monthly = monthly_data[column].min()
        max_monthly_date = monthly_data[column].idxmax()
        min_monthly_date = monthly_data[column].idxmin()

        # Print the results
        print(f'\n{column}: Max Weekly {max_weekly} on {max_weekly_date}')
        print(f'{column}: Min Weekly {min_weekly} on {min_weekly_date}')
        print(f'{column}: Max Monthly {max_monthly} on {max_monthly_date}')
        print(f'{column}: Min Monthly {min_monthly} on {min_monthly_date}')