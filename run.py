import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from linearmodels.panel import RandomEffects
from linearmodels import PanelOLS
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_palette("colorblind")

def prepare_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Convert data types
    df['TIME_PERIOD'] = pd.to_numeric(df['TIME_PERIOD'], errors='coerce')
    df['OBS_VALUE'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
    
    # Extract data by indicator
    gdp_growth = df[df['INDICATOR'] == 'Gross domestic product (GDP), Constant prices, Percent change'].copy()
    exports_growth = df[df['INDICATOR'] == 'Exports of goods and services, Volume, Free on board (FOB), Percent change'].copy()
    expenditure = df[df['INDICATOR'] == 'Expenditure, General government, Percent of GDP'].copy()
    revenue = df[df['INDICATOR'] == 'Revenue, General government, Percent of GDP'].copy()
    cpi = df[df['INDICATOR'] == 'All Items, Consumer price index (CPI), Period average, percent change'].copy()
    
    # Create pivot tables
    gdp_pivot = gdp_growth.pivot_table(values='OBS_VALUE', index=['COUNTRY', 'TIME_PERIOD'])
    exports_pivot = exports_growth.pivot_table(values='OBS_VALUE', index=['COUNTRY', 'TIME_PERIOD'])
    expenditure_pivot = expenditure.pivot_table(values='OBS_VALUE', index=['COUNTRY', 'TIME_PERIOD'])
    revenue_pivot = revenue.pivot_table(values='OBS_VALUE', index=['COUNTRY', 'TIME_PERIOD'])
    cpi_pivot = cpi.pivot_table(values='OBS_VALUE', index=['COUNTRY', 'TIME_PERIOD'])
    
    # Reset indices and rename columns to be more descriptive
    gdp_pivot_reset = gdp_pivot.reset_index().rename(columns={'OBS_VALUE': 'GDP_Growth'})
    exports_pivot_reset = exports_pivot.reset_index().rename(columns={'OBS_VALUE': 'Exports_Growth'})
    expenditure_pivot_reset = expenditure_pivot.reset_index().rename(columns={'OBS_VALUE': 'Expenditure'})
    revenue_pivot_reset = revenue_pivot.reset_index().rename(columns={'OBS_VALUE': 'Revenue'})
    cpi_pivot_reset = cpi_pivot.reset_index().rename(columns={'OBS_VALUE': 'CPI'})
    
    # Merge GDP and exports to calculate EPI
    epi_data = pd.merge(gdp_pivot_reset, exports_pivot_reset, on=['COUNTRY', 'TIME_PERIOD'])
    epi_data['EPI'] = 0.7 * epi_data['GDP_Growth'] + 0.3 * epi_data['Exports_Growth']
    
    # Merge remaining data
    data = pd.merge(epi_data, expenditure_pivot_reset, on=['COUNTRY', 'TIME_PERIOD'])
    data = pd.merge(data, revenue_pivot_reset, on=['COUNTRY', 'TIME_PERIOD'])
    data = pd.merge(data, cpi_pivot_reset, on=['COUNTRY', 'TIME_PERIOD'])
    
    return data

def descriptive_analysis(data):
    # Check if data already has MultiIndex structure
    if not isinstance(data.index, pd.MultiIndex):
        # If not, ensure COUNTRY and TIME_PERIOD are set as index
        if 'COUNTRY' in data.columns and 'TIME_PERIOD' in data.columns:
            data = data.set_index(['COUNTRY', 'TIME_PERIOD'])
    
    # Make a copy with reset index for summary statistics
    data_reset = data.reset_index()
    
    # Summary statistics
    print("Summary Statistics:")
    print(data_reset.describe())
    
    # Get unique countries from the data
    countries = data.index.get_level_values(0).unique()
    print(f"Countries in dataset: {', '.join(countries)}")
    
    # Time series plots for Argentina (or first country if Argentina not available)
    target_country = 'Argentina' if 'Argentina' in countries else countries[0]
    
    # Get country data using xs for safer MultiIndex access
    country_data = data.xs(target_country, level=0).reset_index()
    
    plt.figure(figsize=(15, 8))
    plt.plot(country_data['TIME_PERIOD'], country_data['Expenditure'], 'teal', linewidth=2, 
             label='Expenditure, % of GDP')
    plt.plot(country_data['TIME_PERIOD'], country_data['Revenue'], 'orange', linewidth=2, 
             label='Revenue, % of GDP')
    plt.plot(country_data['TIME_PERIOD'], country_data['CPI'], 'red', linewidth=2, 
             label='CPI, % change')
    plt.plot(country_data['TIME_PERIOD'], country_data['GDP_Growth'], 'g-', linewidth=1, 
             label='GDP Growth, %')
    plt.plot(country_data['TIME_PERIOD'], country_data['Exports_Growth'], 'y-', linewidth=1, 
             label='Exports Growth, %')
    plt.plot(country_data['TIME_PERIOD'], country_data['EPI'], 'r--', linewidth=2, 
             label='Economic Performance Index')
    
    plt.title(f'Economic Indicators Over Time ({target_country})', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{target_country}_time_series.png')
    plt.show()
    
    # Scatter plots with regression lines
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with regression line for Expenditure vs EPI
    sns.regplot(data=data_reset, x='Expenditure', y='EPI', scatter=True, 
                color='teal', label='EPI vs Expenditure', scatter_kws={'alpha':0.5})
    
    # Add another scatter plot with regression line for Revenue vs EPI
    sns.regplot(data=data_reset, x='Revenue', y='EPI', scatter=True, 
                color='orange', label='EPI vs Revenue', scatter_kws={'alpha':0.5})
    
    plt.title('EPI vs Government Finance', fontsize=16)
    plt.xlabel('Percent of GDP', fontsize=14)
    plt.ylabel('Economic Performance Index (EPI)', fontsize=14)
    
    # Add a custom legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='teal', lw=2),
                   Line2D([0], [0], color='orange', lw=2)]
    plt.legend(custom_lines, ['EPI vs Expenditure', 'EPI vs Revenue'], fontsize=12)
    
    plt.tight_layout()
    plt.savefig('epi_vs_govt_finance.png')
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr_data = data.copy()
    corr_matrix = corr_data.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                linewidths=0.5, fmt='.3f', vmin=-1, vmax=1)
    plt.title('Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.show()
    
    # Growth rates analysis for each country
    for country in countries:
        try:
            # Get data for the specific country
            country_data = data.xs(country, level=0).reset_index()
            
            if len(country_data) >= 2:  # Need at least 2 data points for growth rate
                # Calculate growth rates
                country_data['Expenditure_Growth'] = country_data['Expenditure'].pct_change() * 100
                country_data['Revenue_Growth'] = country_data['Revenue'].pct_change() * 100
                
                # Calculate mean growth rates
                mean_exp_growth = country_data['Expenditure_Growth'].mean()
                mean_rev_growth = country_data['Revenue_Growth'].mean()
                
                # Plot fiscal indicators growth rates
                plt.figure(figsize=(12, 6))
                plt.plot(country_data['TIME_PERIOD'][1:], country_data['Expenditure_Growth'][1:], 
                         'b-', linewidth=2, label='Expenditure Growth Rate')
                plt.plot(country_data['TIME_PERIOD'][1:], country_data['Revenue_Growth'][1:], 
                         'g-', linewidth=2, label='Revenue Growth Rate')
                
                # Add horizontal lines for the means
                plt.axhline(y=mean_exp_growth, color='b', linestyle='--', alpha=0.5, 
                           label=f'Avg. Expenditure Growth: {mean_exp_growth:.2f}%')
                plt.axhline(y=mean_rev_growth, color='g', linestyle='--', alpha=0.5, 
                           label=f'Avg. Revenue Growth: {mean_rev_growth:.2f}%')
                
                plt.title(f'{country}: Fiscal Indicators Growth Rates', fontsize=14)
                plt.xlabel('Year', fontsize=12)
                plt.ylabel('Percent Change (%)', fontsize=12)
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'{country}_growth_rates.png')
                plt.close()  # Close to avoid too many open figures
                
                # Print summary statistics for the country
                print(f"\nGrowth Rate Summary for {country}:")
                print(f"Average Expenditure Growth Rate: {mean_exp_growth:.2f}%")
                print(f"Average Revenue Growth Rate: {mean_rev_growth:.2f}%")
                print(f"Correlation between Expenditure and EPI: {country_data['Expenditure'].corr(country_data['EPI']):.3f}")
                print(f"Correlation between Revenue and EPI: {country_data['Revenue'].corr(country_data['EPI']):.3f}")
        except Exception as e:
            print(f"Error processing data for {country}: {e}")
    
    # Create a panel of trend plots for top countries
    top_countries = countries[:min(4, len(countries))]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    axes = axes.flatten()
    
    for i, country in enumerate(top_countries):
        if i < len(axes):
            country_data = data.xs(country, level=0).reset_index()
            axes[i].plot(country_data['TIME_PERIOD'], country_data['EPI'], 'r-', linewidth=2)
            axes[i].set_title(f'{country}: EPI Trend', fontsize=12)
            axes[i].grid(True)
            
            # Add polynomial trendline
            z = np.polyfit(country_data['TIME_PERIOD'], country_data['EPI'], 2)
            p = np.poly1d(z)
            axes[i].plot(country_data['TIME_PERIOD'], p(country_data['TIME_PERIOD']), 'b--', 
                        linewidth=1, label='Trend')
            axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('country_epi_trends.png')
    plt.show()


def run_panel_models(data):
    if not isinstance(data.index, pd.MultiIndex) or len(data.index.levels) != 2:
        data = data.reset_index()
        data = data.set_index(['COUNTRY', 'TIME_PERIOD'])
    
    Y = data['EPI']
    X = data[['Expenditure', 'Revenue', 'CPI']]
    X = sm.add_constant(X)
    
    # Run the Fixed Effects model
    fixed_effects_model = PanelOLS(Y, X, entity_effects=True)
    fixed_effects_results = fixed_effects_model.fit()
    
    print("\nFixed Effects Model Results:")
    print(fixed_effects_results)
    
    # Compare with other models using Hausman test
    random_effects_model = RandomEffects(dependent=Y, exog=X)
    random_effects_results = random_effects_model.fit()
    
    pooled_ols_model = PanelOLS(Y, X)
    pooled_ols_results = pooled_ols_model.fit()
    
    # Hausman test
    from linearmodels.panel import compare
    hausman_test = compare({'Fixed Effects': fixed_effects_results, 
                           'Random Effects': random_effects_results})
    
    print("\nModel Comparison (Hausman Test):")
    print(hausman_test)
    
    coefs = fixed_effects_results.params
    coefs_df = pd.DataFrame({'Variable': coefs.index, 'Coefficient': coefs.values})
    coefs_df = coefs_df[coefs_df['Variable'] != 'const']  # Exclude constant
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Variable', y='Coefficient', data=coefs_df)
    plt.title('Coefficients from Fixed Effects Model', fontsize=16)
    plt.xlabel('Variable', fontsize=14)
    plt.ylabel('Coefficient Value', fontsize=14)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.tight_layout()
    plt.savefig('coefficients_plot.png')
    plt.show()
    
    # Residual diagnostics
    residuals = fixed_effects_results.resids
    fitted_values = Y - residuals
    
    plt.figure(figsize=(12, 8))
    plt.scatter(fitted_values, residuals)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Residuals vs Fitted Values (Fixed Effects Model)', fontsize=16)
    plt.xlabel('Fitted Values', fontsize=14)
    plt.ylabel('Residuals', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('residuals_plot.png')
    plt.show()
    
    return fixed_effects_results

def main(file_path):
    print("Loading and preparing data...")
    data = prepare_data(file_path)
    
    print("\nPerforming descriptive analysis...")
    descriptive_analysis(data)
    
    print("\nEstimating panel data models...")
    model_results = run_panel_models(data)
    
    print("\nAnalysis complete!")
    return data, model_results

if __name__ == "__main__":
    data, model_results = main('dataset2.csv')
