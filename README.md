# Plotting COVID-19 number of deaths and cases in NCR

Python file used in plotting a bar plot and scatter plot of the total number of COVID-19 deaths and cases in the National Capital Region (NCR) per day. You can access recent data from the Department of Health (DOH) through this link: https://bit.ly/DataDropPH (dataset available at the link at the end of the pdf file).

NOTES: 
- Data used for this project is from the July 3, 2020 data drop. Thus, dates included are from before the start of quarantine until July 3. Just replace the csv file in the path to a recent dataset. 
- The plot is separated by quarantine status. For recent dataset, adjust lines 40-48 to include new quarantine status.

WHAT I INITIALLY DID:
- plot the total number of deaths and cases per day using a bar plot and/or scatter plot with different quarantine statuses categorized
- plot age distribution of confirmed cases and deaths
- plot healthstatus column against sex
- use scipy's chisquare to calculate for the chi-square goodness-of-fit value of the distribution 
- use scipy's chi2_contingency to test the independence of the healthstatus versus sex plot
