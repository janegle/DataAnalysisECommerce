import pickle
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

KPI = ['ATC', 'CVR', 'Clicks', 'SessionRevenue']
PATH_RESULTS = "./results/"
PATH_IMAGES = "./images/"


# Summarize the loaded data
def summarize_data(df: pd.DataFrame):
    row = df.shape[0]
    col = df.shape[1]

    print("################################################################\n")
    print("Summary for the loaded dataset\n")
    print("The total number of rows is {}\n".format(row))
    print("The total number of columns is {}\n".format(col))
    print("Descriptive Statistics:\n\n{}\n".format(df.describe()))
    print(format(df.info()))

    for i in df.columns:
        count_df = df.groupby(by=i)[i].count()
        print("Count summary for {}:".format(i))
        print(count_df)
        print("################################################################\n")


# Check count of missing data for each variable
def print_count_null(df: pd.DataFrame):
    print("Count of null values:\n")
    for i in df.columns:
        print(str(i) + " " + str(df[i].isnull().sum()))


# Check randomization
def check_randomization(df: pd.DataFrame, feature: str):
    print(pd.crosstab(df[feature], df['testGroup']).apply(lambda r: r / r.sum(), axis=0))


def t_test_results(df: pd.DataFrame, file_name: str):
    """
    Calculate key data points (per generate_stats function) for each given KPI by various features:
    Input:
    - df: pandas dataframe
    Output:
    - csv file saved in folder "results"
    """
    features = ['platform', 'visitorType', 'CategoryID']
    dfs = []
    for feature in features:
        for type in df[feature].unique():
            df_temp = cut_by_feature(df, feature, type, 'equal')
            for kpi in KPI:
                result = generate_stats(df_temp, kpi, type)
                dfs.append(result)
    result = pd.DataFrame(dfs)
    result.to_csv(PATH_RESULTS + file_name, index=False)


def t_test_results_cum(df: pd.DataFrame, file_name: str):
    """
    Calculate key data points (per generate_stats function) for each given KPI by date cumulatively:
    Input:
    - df: pandas dataframe
    Output:
    - csv file saved in folder "results"
    """
    features = ['SessionStartDate']

    for feature in features:
        dfs = []
        for date in sorted(df[feature].unique()):
            df_temp = cut_by_feature(df, feature, date, 'smaller')
            for kpi in KPI:
                result = generate_stats(df_temp, kpi, date)
                dfs.append(result)
        result = pd.DataFrame(dfs)
        result.to_csv(PATH_RESULTS + file_name, index=False)


def generate_stats(df: pd.DataFrame, kpi: str, feature: str):
    """
    Calculate the following data points for each KPI:
    - mean of the control group
    - mean of the variation group
    - lift
    - t stat
    - p value
    Methodology: two-sample unpooled t-test with unequal variances (which would give the same results
    as two-proportion unpooled z-test with unequal variances where the outcome is binary
    Input:
    - df: pandas dataframe
    - kpi: string
    - feature: string
    Output:
    - dictionary containing the data points
    """
    df_control = df.loc[df['testGroup'] == 'control']
    df_variation = df.loc[df['testGroup'] == 'variation']

    df1 = df_control[kpi]
    df2 = df_variation[kpi]
    t, p = stats.ttest_ind(df2, df1, equal_var=False)
    lift = (df2.mean() - df1.mean()) / df1.mean()

    return {
        'Feature': feature,
        'KPI': kpi,
        'mean_control': df1.mean(),
        'mean_variation': df2.mean(),
        'lift': lift,
        't-stat': t,
        'p-value': p
    }


# Filter a dataframe by a given feature and sign
def cut_by_feature(df: pd.DataFrame, feature: str, value: str, sign: str):
    if sign == 'equal':
        df = df.loc[df[feature] == value]
    elif sign == 'greater':
        df = df.loc[df[feature] >= value]
    elif sign == 'smaller':
        df = df.loc[df[feature] <= value]
    return df


# Plot graphs of lift or t-stat over time for each KPI
def plot_graph(df: pd.DataFrame, plot_type: str, kpi: str, output_type: str):
    df = df.rename(columns={'Feature': 'Date'})
    df = df[['Date', 'KPI', 'lift', 't-stat']]
    df = df.loc[df['KPI'] == kpi]
    plt.figure(figsize=(15, 8))
    graph = sns.lineplot(data=df, x='Date', y=plot_type)
    if plot_type == 'lift':
        graph.yaxis.set_major_formatter(PercentFormatter(1))
    graph.set_title(kpi + " Trends for Mobile Users")
    plt.xticks(rotation=15)

    if output_type == 'save':
        plt.savefig(PATH_IMAGES + kpi + '_' + plot_type + '.png')
    else:
        plt.show()


def main():
    df = pd.read_csv('./data/ECommerceDataSet.csv')
    print("\nThe data is successfully loaded!\n")
    summarize_data(df)
    print_count_null(df)
    # Remove outliers - 11 rows have SessionRevenue == $500,000 and all in variation group
    df_check1 = df[df['SessionRevenue'] > 100000]
    print("No of rows where SessionRevenue > $100,000: " + str(df_check1.shape[0]))
    print("Descriptive Statistics where SessionRevenue > $100,000:\n\n{}\n".format(df_check1.describe()))
    df_check2 = df[(df['SessionRevenue'] > 10000) & (df['SessionRevenue'] < 100000)]
    print("No of rows where SessionRevenue is between $10,000 and $100,000: " + str(df_check2.shape[0]))
    print("Descriptive Statistics where SessionRevenue is between $10,000 and $100,000:\n\n{}\n".format(
        df_check2.describe()))
    df = df[df['SessionRevenue'] != 500000]
    # Check for bad data: is there any session where CVR == 1 but Clicks == 0 or ATC == 0
    df_check3 = df[(df['CVR'] == 1) & ((df['Clicks'] == 0) | (df['ATC'] == 0))]
    print("No of rows where CVR is 1 but Clicks is 0 or ATC is 0: " + str(df_check3.shape[0]))
    # Check randomization
    check_randomization(df, 'platform')
    check_randomization(df, 'visitorType')
    check_randomization(df, 'SessionStartDate')
    check_randomization(df, 'CategoryID')
    # Run statistical tests
    t_test_results(df, 't_test_results_all.csv')
    t_test_results_cum(df, 't_test_results_all_cumulative.csv')
    # Run further statistical tests for mobile and desktop
    df_mobile = df[df['platform'] == 'Mobile Site']
    t_test_results_cum(df_mobile, 't_test_results_mobile_cumulative.csv')
    df_desktop = df[df['platform'] == 'Desktop']
    t_test_results_cum(df_desktop, 't_test_results_desktop_cumulative.csv')
    # Plot graph of lift over time for each KPI among mobile users
    df_t_test_results = pd.read_csv('./results/t_test_results_mobile_cumulative.csv')
    plot_graph(df_t_test_results, 'lift', 'SessionRevenue', 'save')
    plot_graph(df_t_test_results, 'lift', 'CVR', 'save')
    plot_graph(df_t_test_results, 'lift', 'ATC', 'save')
    plot_graph(df_t_test_results, 't-stat', 'SessionRevenue', 'save')
    plot_graph(df_t_test_results, 't-stat', 'CVR', 'save')
    plot_graph(df_t_test_results, 't-stat', 'ATC', 'save')


if __name__ == "__main__":
    main()
