Introduction
This Tableau file contains visualizations and analysis of turnover data by region and industry. The purpose of this project is to explore the relationship between turnover, regions, and industries based on the provided datasets.

Datasets
1. "50 repeated_name_with_post.csv": This CSV file contains turnover data for differentregions. It includes columns such as region name, Postcode, Year, 'Sum of 
$0k to less than $50k' which means the business count number, and LGA code.
2. "v2 Turnover Regions and industries from 2020-2022.xlsx": This Excel file provides additional information about turnover, regions, and industries. It includes multiple 
sheets, and we will be using "Sheet4" for our analysis. This sheet contains the LGA code and other relevant details such as highly impacted index, change rate, etc.

Instructions for Opening the File
1. Download the Tableau file "turnover by region and industry - final.twb" from the provided location.
2. Launch Tableau Desktop, ensuring you have a compatible version.
3. Select "Open" from the Tableau Desktop home screen.
4. Browse to the location where you saved the downloaded Tableau file, and select it to open.

Adding Data Sources
1. After opening the Tableau file, navigate to the "Data" tab at the top of the Tableau Desktop window.
2. In the "Connections" pane on the left, click on the "Add" button to add data sources.
3. Locate and select the "50 repeated_name_with_post.csv" file from your local system.
4. Repeat the same steps to add the "v2 Turnover Regions and industries from 2020-2022.xlsx" file.
5. Make sure to choose the correct sheet ("Sheet4") from the Excel file when prompted.

Data Connection and Joins
1. In the "Data" tab, you will see the added data sources listed in the "Connections" pane.
2. To establish a connection between the two data sources, we need to perform a join based on the "LGA Code" column.
3. Drag and drop the "LGA Code" column from the CSV data source to the "LGA Code" column in the Excel data source.
4. Choose the appropriate join type (e.g., inner join) to ensure the data is connected based on the matching LGA codes.

And then, you could check every worksheet and dashboard from our .twb file.