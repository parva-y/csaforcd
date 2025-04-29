import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_competitor_data(start_date='2024-09-01', num_days=30):
    """
    Generate sample competitor search data with the same schema
    """
    # Create date range
    dates = [datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=i) for i in range(num_days)]
    
    # Create dataframe with dates
    df = pd.DataFrame({'Date': dates})
    
    # Define competitors
    competitors = [
        'Angel One', 'Binance', 'CoinDCX', 'Coinswitch', 'Delta Exchange', 
        'Dhan', 'Groww', 'Lemonn', 'Upstox', 'WazirX', 'Zerodha', 'Mudrex'
    ]
    
    # Define search terms
    search_terms = [
        'Cryptocurrency', 'Mutual fund', 'Stock Market', 'Bitcoin'
    ]
    
    # Add columns for competitors with random search volumes
    for comp in competitors:
        # Base value that increases steadily
        base = random.randint(1000, 5000)
        # Generate search volumes with some randomness
        df[comp] = [base + int(np.random.normal(i*10, 200)) for i in range(num_days)]
        # Ensure no negative values
        df[comp] = df[comp].apply(lambda x: max(0, x))
    
    # Add columns for search terms
    for term in search_terms:
        # Higher base values for general search terms
        base = random.randint(8000, 15000)
        # Generate search volumes with some randomness
        df[term] = [base + int(np.random.normal(i*15, 500)) for i in range(num_days)]
        # Ensure no negative values
        df[term] = df[term].apply(lambda x: max(0, x))
    
    # Add CoinDCX installs column
    df['Installs - DCX'] = [random.randint(4000, 6000) for _ in range(num_days)]
    
    # Format dates to match the example format (dd/mm/yyyy)
    df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')
    
    return df

def generate_campaign_data(start_date='2024-10-01', num_days=15):
    """
    Generate sample campaign data with the same schema
    """
    # Create date range
    dates = [datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=i) for i in range(num_days)]
    
    # First section: Top Impact/Roadblock
    header_row = pd.DataFrame({
        0: ['Top Impact/ Roadblock'],
        1: ['Top ROS'],
        2: ['Date'],
        3: ['All Impressions'],
        4: ['All Spends'],
    })
    
    # Add platform columns to the header
    platforms = [
        'Moneycontrol', 'ET Now', 'Good Returns', 'HT', 'ET', 
        'Vi', 'MIQ', 'ET', 'Good Returns', 'Dailyhunt', 'Moneycontrol', 'PayTM'
    ]
    
    for platform in platforms:
        header_row[len(header_row.columns)] = [f"{platform} Imp"]
        header_row[len(header_row.columns)] = [f"{platform} Spends"]
    
    # Add an empty row
    empty_row = pd.DataFrame([[''] * len(header_row.columns)])
    
    # Second section: Date and metrics
    metrics_header = pd.DataFrame({
        0: ['Date'],
        1: ['Impressions'],
        2: ['Spends']
    })
    
    # Generate daily metrics
    metrics_data = []
    for date in dates:
        # For the first 5 days, no campaign activity
        if date < datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=5):
            impressions = 0
            spends = 0
        else:
            # Random impressions between 100,000 and 700,000
            impressions = random.randint(100000, 700000)
            # Random spend between 10,000 and 90,000
            spends = random.randint(10000, 90000)
        
        # Format with commas
        impressions_str = f"{impressions:,}" if impressions > 0 else ""
        spends_str = f"{spends:,}" if spends > 0 else ""
        
        metrics_data.append([date.strftime('%d/%m/%Y'), impressions_str, spends_str])
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Combine all sections to create the final dataframe
    # We'll use a workaround for the complex structure
    
    # First, save all sections to CSV
    header_row.to_csv('temp_header.csv', index=False, header=False)
    empty_row.to_csv('temp_empty.csv', index=False, header=False)
    metrics_header.to_csv('temp_metrics_header.csv', index=False, header=False)
    metrics_df.to_csv('temp_metrics.csv', index=False, header=False)
    
    # Now read them back with the right settings
    import os
    
    with open('temp_header.csv', 'r') as f1, \
         open('temp_empty.csv', 'r') as f2, \
         open('temp_metrics_header.csv', 'r') as f3, \
         open('temp_metrics.csv', 'r') as f4, \
         open('temp_combined.csv', 'w') as out:
        out.write(f1.read())
        out.write(f2.read())
        out.write(f3.read())
        out.write(f4.read())
    
    combined_df = pd.read_csv('temp_combined.csv', header=None)
    
    # Clean up temporary files
    for file in ['temp_header.csv', 'temp_empty.csv', 'temp_metrics_header.csv', 'temp_metrics.csv', 'temp_combined.csv']:
        if os.path.exists(file):
            os.remove(file)
    
    return combined_df

if __name__ == "__main__":
    # Generate sample data
    print("Generating sample competitor data...")
    competitors_df = generate_competitor_data()
    
    print("Generating sample campaign data...")
    campaign_df = generate_campaign_data()
    
    # Save to CSV
    competitors_df.to_csv('sample_competitor_data.csv', index=False)
    campaign_df.to_csv('sample_campaign_data.csv', index=False, header=False)
    
    print("Sample data generated:")
    print("1. sample_competitor_data.csv")
    print("2. sample_campaign_data.csv")
    
    try:
        # Try to save as Excel if openpyxl is available
        competitors_df.to_excel('sample_competitor_data.xlsx', index=False)
        campaign_df.to_excel('sample_campaign_data.xlsx', index=False, header=False)
        print("3. sample_competitor_data.xlsx")
        print("4. sample_campaign_data.xlsx")
    except Exception as e:
        print(f"Could not save as Excel: {e}")
        print("Install openpyxl to enable Excel output: pip install openpyxl")
