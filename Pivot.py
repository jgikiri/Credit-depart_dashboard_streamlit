# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import os
import calendar
from datetime import datetime

# page set up
st.set_page_config(page_title="Credit Dashboard",
                   page_icon="chart_with_upwards_trend",
                   layout="wide",
                   initial_sidebar_state="collapsed"
)

# Page title
st.title("CREDIT DEPARTMENT DASHBOARD")


# Loading default dataset
path = os.path.dirname(__file__)
path = os.path.join(path, "loan_listing.xlsx")

# Side bar
st.sidebar.header("CREDIT DEPART")
data = st.sidebar.file_uploader("Upload Data", type = ["xlsx", "csv"])

# Loading datasets
def load_data (file):
    df = pd.read_excel(file)
    df["Days In Arrears"] = pd.to_numeric(df["Days In Arrears"])
    df["Expected End Date"] = pd.to_datetime(df["Expected End Date"])
    df["Last Payment Date"] = pd.to_datetime(df["Last Payment Date"])
    df["Date Due"] = pd.to_datetime(df["Date Due"])
    df["Date Disbursed"] = pd.to_datetime(df["Date Disbursed"])
    return df


if data is not None:
    df = load_data(data)
else:
    df = load_data(path)

# Add select boxes for different func
menu = ["Snapshot", "Analysis(Disbursement)", "Trend Analysis PAR"]
selection = st.sidebar.selectbox("Key Indicators", menu)

st.sidebar.write("Key analysis for reporting. Takes loan listing_V2 table Nav and auromatically analyses various metrics")


if selection == "Snapshot":
    st.subheader("Data Summary")
    # Check Box
    if st.checkbox("Number of Rows and Columns"):
        st.write("Rows and Columns")
        st.write(df.shape)
    st.dataframe(df.sample(n = 8))
    col1, col2, col3 = st.columns(3)


    with col1:
        st.subheader("Overal Portfolio At Risk, Total Amount In Kes")
        # Create Bins
        bins = [-float('inf'), 1, 30, 90, 360, float('inf')]
        labels = ['Below 1', '1-30', '31-90', '91-360', 'Above 360']
        df2 = df.copy()

        df2['Days In Arrears Category'] = pd.cut(df2["Days In Arrears"], bins=bins, labels=labels)
        # Pivot table
        pivot_table = pd.pivot_table(df2, values='Outstanding Loan', index='Days In Arrears Category', aggfunc='sum')

        # Sum Outstanding Loan
        pivot_table.loc['Total'] = pivot_table.sum()

        # Format the sum figure to a readable format
        pivot_table["Outstanding Loan"] = pivot_table["Outstanding Loan"].map('{:,.2f}'.format)

        st.dataframe(pivot_table)
        # Row2 Col1
        st.subheader("Overal Product OLB, Total Amount In Kes")
        # Pivot table product
        pivot_table = pd.pivot_table(df2, values='Outstanding Loan', index='Product Name', aggfunc='sum')

        # Sum Outstanding Loan
        pivot_table.loc['Total'] = pivot_table.sum()

        # Format the sum figure to a readable format
        pivot_table['Outstanding Loan'] = pivot_table['Outstanding Loan'].map('{:,.2f}'.format)

        st.dataframe(pivot_table)



    # Colum2
    with col2:

        st.subheader("Overal Portfolio At Risk, Total Count")
        # Create Bins
        bins = [-float('inf'), 1, 30, 90, 360, float('inf')]
        labels = ['Below 1', '1-30', '31-90', '91-360', 'Above 360']

        df2["Days In Arrears Category"] = pd.cut(df2['Days In Arrears'], bins=bins, labels=labels)
        # Pivot table
        pivot_table = pd.pivot_table(df2, values='Outstanding Loan', index='Days In Arrears Category', aggfunc='count')

        # Sum Outstanding Loan
        pivot_table.loc['Total'] = pivot_table.sum()

        st.dataframe(pivot_table)

        # Row2 Col2
        # Pivot table product
        st.subheader("Overal Product OLB, Total count")
        pivot_table = pd.pivot_table(df2, values='Outstanding Loan', index='Product Name', aggfunc='count')

        # Sum Outstanding Loan
        pivot_table.loc['Total'] = pivot_table.sum()

        # Format the sum figure to a readable format
        pivot_table['Outstanding Loan'] = pivot_table['Outstanding Loan']

        st.dataframe(pivot_table)

    # Column 3
    with col3:
        st.subheader("PAR in Absoulute and Percentage")
        # Define bins and labels
        bins = [-float('inf'), 1, 30, float('inf')]
        labels = ['Below 1', '1-30', 'Above 30']

        # Create a new column for Days In Arrears Category
        df2["Days In Arrears Category"] = pd.cut(df2['Days In Arrears'], bins=bins, labels=labels, right=False)

        # Pivot table
        pivot_table = pd.pivot_table(df2, values='Outstanding Loan', index='Days In Arrears Category',
                                     aggfunc='sum').reset_index()

        # Remove commas and convert "Outstanding Loan" to float
        pivot_table["Outstanding Loan"] = pivot_table["Outstanding Loan"].replace('[\$,]', '', regex=True).astype(float)

        # Format the "Outstanding Loan" column
        pivot_table["Outstanding Loan"] = pivot_table["Outstanding Loan"].map('{:,.2f}'.format)

        # Add a row for the total sum
        total_outstanding_loan = pivot_table['Outstanding Loan'].replace('[\$,]', '', regex=True).astype(float).sum()
        pivot_table.loc[len(pivot_table.index)] = ['Total', total_outstanding_loan]

        # Add a column with the percentage
        pivot_table['Percentage of Total'] = (pivot_table['Outstanding Loan'].replace('[\$,]', '', regex=True).astype(
            float) / total_outstanding_loan) * 100

        # Format the "Percentage of Total" column
        pivot_table['Percentage of Total'] = pivot_table['Percentage of Total'].map('{:.2f}%'.format)

        # Display the resulting pivot table
        st.dataframe(pivot_table)


elif selection == "Analysis(Disbursement)":

    if st.checkbox("Descriptive Statistics"):
        st.write(df.describe())

        # Get the current year and month
    current_year = datetime.now().year
    current_month = datetime.now().month

    # Create a list of numerical months (1 = January, 2 = February, etc.) and years
    months = list(range(1 ,13))
    years = list(range(current_year - 3, current_year + 1))  # Adjust the range as needed

    # Create select boxes for month and year
    selected_month = st.selectbox("Select Month", months, index=current_month - 1)
    selected_year = st.selectbox("Select Year", years,
                                 index=3)  # Set the default index to the current year or any other default

    col6, col7, col8 = st.columns(3)

    # Columns
    with col6:
        df2 = df.copy()
        st.subheader("Total Net Disbursement By Product Amount In Kes")

        # Input =  month and year
        desired_month = selected_month  # Replace with the desired month
        desired_year = selected_year  # Replace with the desired year

        # Filter the DataFrame for the specified month and year
        filtered_df0 = df2[
            (df2["Date Disbursed"].dt.month == desired_month) & (df2["Date Disbursed"].dt.year == desired_year)]

        # Pivot the data
        pivot_table1 = pd.pivot_table(filtered_df0,
                                      values="Net Disbursed Amount",
                                      index="Product Name",
                                      aggfunc="sum"
                                      )

        # Format the sum figure to be more readable
        pivot_table1["Net Disbursed Amount"] = pivot_table1["Net Disbursed Amount"].map("{:,.2f}".format)

        # Print or display the resulting pivot table
        st.dataframe(pivot_table1)

    with col7:
        st.subheader("Total Net Disbursement By Product Amount (Count)")
        # Input =  month and year
        desired_month2 = selected_month  # Replace with the desired month
        desired_year2 = selected_year  # Replace with the desired year

        # Filter the DataFrame for the specified month and year
        filtered_df1 = df2[
            (df2["Date Disbursed"].dt.month == desired_month2) & (df["Date Disbursed"].dt.year == desired_year2)]

        # Pivot the data
        pivot_table = pd.pivot_table(filtered_df1,
                                     values="Net Disbursed Amount",
                                     index="Product Name",
                                     aggfunc="count"
                                     )

        # Format the sum figure to be more readable
        pivot_table["Net Disbursed Amount"] = pivot_table["Net Disbursed Amount"]

        # Print or display the resulting pivot table
        st.dataframe(pivot_table)

    col8 = st.columns(1)[0]
    with col8:
        col8.subheader(f'Disbursement By Product Distribution in {desired_month}/{desired_year}')
        # Input =  month and year
        desired_month3 = selected_month  # Replace with the desired month
        desired_year3 = selected_year  # Replace with the desired year

        # Filter the DataFrame for the specified month and year
        filtered_df3 = df2[
            (df2["Date Disbursed"].dt.month == desired_month3) & (df["Date Disbursed"].dt.year == desired_year3)]

        product = filtered_df3["Product Name"].value_counts()
        label = product.index
        # Ploting a pie chart

        fig = px.pie(product,
                     values=product,
                     names=label,
                     );
        st.plotly_chart(fig)

    col9, col10 = st.columns(2)
    with col9:
        st.subheader(f'Disbursement Count by Month in {desired_year}')

        # Input = year
        desired_year = selected_year  # Replace with the desired year

        # Filter the DataFrame for the specified month and year
        filtered_df4 = df2[(df2["Date Disbursed"].dt.year == desired_year)]

        # Add month when loan was disbursed
        filtered_df4["Month Disbursed"] = pd.DatetimeIndex(filtered_df4["Date Disbursed"]).month

        # Plotting Month Vs Disbursement count

        monthly_counts = filtered_df4['Month Disbursed'].value_counts().sort_index()
        monthly_sums = filtered_df4.groupby('Month Disbursed')['Net Disbursed Amount'].sum() / 1000

        months = monthly_counts.index

        # Create a Plotly bar chart
        fig = px.bar(
            x=months,
            y=monthly_counts,
            labels={'x': 'Months', 'y': 'Disbursement Count'},
        )

        # Set layout parameters
        fig.update_layout(
            xaxis=dict(tickmode='array', tickvals=months, ticktext=[str(i) for i in months]),
            xaxis_title='Months',
            yaxis_title='Disbursement Count',
            showlegend=False,
        )

        # Display the Plotly chart in Streamlit
        st.plotly_chart(fig)

    with col10:
        st.subheader(f'Total Disbursement by Month in {desired_year}, Amount in Kes')

        # Input = year
        desired_year = selected_year  # Replace with the desired year

        # Filter the DataFrame for the specified month and year
        filtered_df4 = df2[(df2["Date Disbursed"].dt.year == desired_year)]

        # Add month when loan was disbursed
        filtered_df4["Month Disbursed"] = pd.DatetimeIndex(filtered_df4["Date Disbursed"]).month

        # Plotting Month Vs Disbursement count

        monthly_counts = filtered_df4['Month Disbursed'].value_counts().sort_index()
        monthly_sums = filtered_df4.groupby('Month Disbursed')['Net Disbursed Amount'].sum() / 1000

        months = monthly_counts.index

        # Create a Plotly bar chart
        fig = px.bar(
            x=months,
            y=monthly_counts,
            labels={'x': 'Months', 'y': 'Disbursement Vol in "000"'},
            color_discrete_sequence=['green'],  # Set the bar color to green
        )

        # Set layout parameters
        fig.update_layout(
            xaxis=dict(tickmode='array', tickvals=months, ticktext=[str(i) for i in months]),
            xaxis_title='Months',
            yaxis_title='Disbursement Vol in "000"',
            showlegend=False,
        )

        # Display the Plotly chart in Streamlit
        st.plotly_chart(fig)

elif selection == "Trend Analysis PAR":
    # Get the current year and month
    current_year = datetime.now().year


    # Create a list of numerical months (1 = January, 2 = February, etc.) and years

    years = list(range(current_year - 3, current_year + 1))  # Adjust the range as needed

    # Create select boxes for month and year

    selected_year = st.selectbox("Select Year", years, index=3)  # Set the default index to the current year or any other default

    col11 = st.columns(1)[0]

    ## Columns
    with col11:
        df2 = df.copy()
        st.subheader("Outstanding Loan by Days In Arrears Category")
        # Input = year
        desired_year4 = selected_year  # Replace with the desired year

        # Filter the DataFrame for the specified year
        filtered_df4 = df2[df2["Date Disbursed"].dt.year == desired_year4]

        # Add month when the loan was disbursed
        filtered_df4["Month Disbursed"] = pd.DatetimeIndex(filtered_df4["Date Disbursed"]).month

        # Define bins and labels
        bins = [-float('inf'), 1, 30, 90, 360, float('inf')]
        labels = ['Below 1', '1-30', '31-90', '91-360', 'Above 360']

        # Create a new column for Days In Arrears Category
        filtered_df4["Days In Arrears Category"] = pd.cut(filtered_df4['Days In Arrears'], bins=bins, labels=labels,
                                                          right=False)

        # Pivot table
        pivot_table = pd.pivot_table(filtered_df4, values='Outstanding Loan',
                                     index=['Month Disbursed', 'Days In Arrears Category'],
                                     aggfunc='sum').reset_index()

        # Create a line graph using Plotly Express
        fig = px.line(
            pivot_table,
            x='Month Disbursed',
            y='Outstanding Loan',
            color='Days In Arrears Category',
            labels={'Month Disbursed': 'Month Disbursed', 'Outstanding Loan': 'Sum of Outstanding Loan'},
            line_shape='linear',
            markers=True,
        )
        # Show the plot
        st.plotly_chart(fig)