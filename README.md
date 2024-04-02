# clustering-apriori
Analysis of Unicorn companies using Clustering &amp; Apriori
# Unicorn Companies
A unicorn company is a privately held company with a current valuation of over USD 1 billion. The main dataset is loaded from PowerBI, this is a merged, transformed and cleaned version of all datasets we used in our analysis. This dataset also includes Unicorns, GDP per capita, GII score, the year the Unicorn was founded, total fund raised, and Number of Patent by Country.

from google.colab import drive
drive.mount('/content/drive')

#install and import packages
!pip install plotly
!pip install pycountry
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pycountry

#load data into collab
unidata = pd.read_csv("")
unidata.head()

# 1) Explatory Data Analysis and Plotly Visualisation



# install and import necessary packages:
import pandas as pd
import numpy as np
import geopandas
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (12.75,8.5)

# upload data from GGDrive, file name: Unicorn CBInsight Fulllist Merged - Final.csv
data = pd.read_csv("Unicorns.csv")
data.head()

data.info() # to gather the datatypes in our dataset.

## 1.1) Overview of the Data

### 1.1.1) Treemap distribution of the unicorn by industry and country


# Overview of the data
fig = px.treemap(unidata,path= ["Country","Industry", "Company"],
            values="Valuation ($B)", color_discrete_sequence=px.colors.qualitative.Pastel)

fig.update_traces(root_color="lightgrey")
fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))

The main aim of displaying a treemap in this context is to provide a visual representation of the distribution of unicorn companies based on their Country, Industry and the Unicorn's valuations.


### 1.1.2) Evaluating 15 Countries by Valuation and Number of Unicorn companies:

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# First plot to show the total Unicorn valuations of the top 15 countries
fig1 = px.bar(data.groupby("Country")["Valuation ($B)"].sum().sort_values(ascending=False)[:15],
              y='Valuation ($B)',
              labels={'Valuation ($B)': 'Valuation ($B)'},
              title='Top 15 Countries by Valuation',
              )

# Second plot is based on the number of unicorns
fig2 = px.bar(pd.DataFrame(data["Country"].value_counts()[:15]).reset_index(),
              x='index',
              y='Country',
              labels={'index': 'Country', 'Country': 'Count'},
              title='Top 15 Countries by Number of Unicorn Companies',
              )

# Create subplot to merge the two plots together
fig = make_subplots(rows=1, cols=2, subplot_titles=('Valuation ($B)', 'Count'))

# Adding plots to the subplot
fig.add_trace(fig1['data'][0], row=1, col=1)
fig.add_trace(fig2['data'][0], row=1, col=2)

# Update layout of the subplot
fig.update_layout(title_text='Top 15 Countries by Valuation and Number of Unicorn Companies')

# Showing the interactive plot
fig.show()


We can clearly see that United States leads both graphs in terms of the Valuation and Number of Unicorns, followed by China in the second place.
Even though India has 59 Unicorns and United Kingdom has 39, India ranks 4th in terms of total valuation and UK ranks 3rd.

###1.1.3) Number of unicorn by Founding Year and Industry

# First plot
fig1 = px.bar(data.groupby("Industry")["Valuation ($B)"].sum().sort_values(ascending=False)[:15],
              y='Valuation ($B)',
              labels={'Valuation ($B)': 'Valuation ($B)'},
              title='Top 15 Industries by Valuation',
              )

# Second plot
fig2 = px.bar(pd.DataFrame(data["Industry"].value_counts()[:15]).reset_index(),
              x='index',
              y='Industry',
              labels={'index': 'Industry', 'Industry': 'Count'},
              title='Top 15 Industries by Number of Unicorn Companies',
              )

# Create subplot
fig = make_subplots(rows=1, cols=2, subplot_titles=('Valuation ($B)', 'Count'))

# Add plots to subplot
fig.add_trace(fig1['data'][0], row=1, col=1)
fig.add_trace(fig2['data'][0], row=1, col=2)

# Update layout
fig.update_layout(title_text='Top 15 Industries by Valuation and Number of Unicorn Companies')

# Show the interactive plot
fig.show()

Fintech ranks the highest in terms of Valuation annd the number of Unicorns. By tapping on the individual bars you could view the valuation and counts of unicorns by industry.

### 1.1.4) Number of Unicorns each Year by Industry (2011-2022)

unidata_with_fyear = unidata[~unidata['Date Joined Year'].isna()] # removing all NA values

num_by_joined_year = unidata_with_fyear["Date Joined Year"].value_counts().reset_index() #counting the founding year of each unicorn
num_by_joined_year["index"] = num_by_joined_year["index"].astype(np.int64)
num_by_joined_year = num_by_joined_year[num_by_joined_year["index"] >= 1990]
num_by_joined_year.sort_values(by=["index"], inplace=True)
years = pd.DataFrame({"years" : num_by_joined_year["index"]})

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=num_by_joined_year["index"], y=num_by_joined_year["Date Joined Year"],
                          mode='lines',
                          name='lines')
              )

updatemenu = []
buttons = []

buttons.append(dict(method='update',
                        label="All industries",
                        visible=True,
                        args=[{'y':[num_by_joined_year["Date Joined Year"]],
                               'x':[num_by_joined_year["index"]],
                               'type':'scatter'},
                              {'title': "Number of unicorns in All industries since 1990"}],
                        )
                  )

for indst in unidata.Industry.unique():
    selected_industry = unidata_with_fyear[unidata_with_fyear["Industry"] == indst]
    temp_vc = selected_industry["Date Joined Year"].value_counts().reset_index()
    temp_vc["index"] = temp_vc["index"].astype(np.int64)
    temp_vc.sort_values(by=["index"], inplace=True)
    result = years.set_index('years').join(temp_vc.set_index('index'), how='left').fillna(0)
    result["Date Joined Year"] = result["Date Joined Year"].astype(int)
    buttons.append(dict(method='update',
                        label=indst,
                        visible=True,
                        args=[{'y':[result["Date Joined Year"]],
                               'x':[years["years"]],
                               'type':'scatter'},
                               {'title': f"Number of unicorns in {indst} since 1990"}],
                        )
                  )

updatemenu = [dict()]

updatemenu[0]['buttons'] = buttons
updatemenu[0]['direction'] = 'down'
updatemenu[0]['showactive'] = True

fig1.update_layout(showlegend=False, updatemenus=updatemenu)
fig1.update_layout(
    title = "Number of unicorns in all industries since 2011",
    xaxis_title='Date Joined Year',
    yaxis_title='Number of Unicorns')
fig1.show()


## 1.2) Assessing the relationship among variables:

import seaborn as sns
import matplotlib.pyplot as plt

# Evaluating the correlation among numeric variables:
corr_matrix = data.corr()
corr_matrix
plt.figure(figsize=(10, 8))  # Set the size of the figure
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap between numeric variables')
plt.show()

This code generates an interactive line plot using Plotly to visualize the number of unicorn companies joined from years 2011 to 2022 in different industries over the years.

# 2) Data Analysis with Unsupervised Machine Learning


## 2.1) Association with Apriori Methods on Investors

#importing relevant packages for apriori
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#separating the 'Select Investors' into separate columns for On-hot encoding
uni_one = data['Select Investors'].str.get_dummies(sep=', ')
uni_one


uni_one.to_csv("Investors.csv")

total_per_column = uni_one.sum()


# Display the top 10 Investors
top_10_columns = total_per_column.nlargest(10)
print(top_10_columns)

uni_table = (uni_one > 0).astype(int) # This one sets all 1+ values to True & convert it to 1
uni_table

#converting the dataframe into True/False format
uni_final=(uni_table > 0)
uni_final

# Determining the minimum support threshold
min_support = 0.002
# Generating frequent itemsets
frequent_itemsets = apriori(uni_final, min_support=min_support, use_colnames=True)
frequent_itemsets.sort_values(by=['support'], ascending=[False])

545 itemsets are generated with a minimum support threshold of 0.002.

min_confidence = 0.5  # Minimum confidence threshold

#Determining rules for apriori with minimum confidence of 0.5
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
rules.sort_values(by=['confidence'], ascending=[False]) #arranging rules based on confidence in descending order

106 rules were generated after setting the minimum confidence threshold to 0.5

# Let us compile a final list with some filter, calculation & sorting
final_rules=rules[(rules['lift']>1)&(rules['support']>=0.002)&(rules['confidence']>=0.5)]
# Determine number of items in X => predicting number of items in Y
final_rules=final_rules.copy() # this step creates an independent list instead of a view on rules from above
# We capture the number of items in the list of each antecedent/consequent set using 'len' function for each row
final_rules['antecedent_count']=final_rules['antecedents'].apply(len)
final_rules['consequent_count']=final_rules['consequents'].apply(len) #X

# Apply some rounding and sorting on Lift and Confidence
final_rules=round(final_rules,3).sort_values(by=['lift', 'confidence'], ascending=[False, False])

# Preserving relevant columns only
final_rules=final_rules[['antecedents', 'antecedent_count', 'consequents', 'consequent_count', 'support', 'confidence', 'lift']]
final_rules

# Loop through the DataFrame - first 10 rows
for index, row in final_rules.head(10).iterrows():
    # Antecedents & Consequents are a special data type called frozenset - we extract it into a list & then a concatenated string
    antecedents = ' & '.join(list(row['antecedents']))
    consequents = ' & '.join(list(row['consequents']))

    # Form a statement using the content from the row to create statements for top 10 rows.
    rule_statement = f"If {antecedents} invests in a company, then {consequents} is {row['confidence'] * 100:.2f}% likely to invest - this is {row['lift']:.2f} times more likely than random chance."

# Print the statements for top 10 rules
    print(rule_statement)

#Visualising apriori results for top 10 rules
import networkx as nx
import matplotlib.pyplot as plt

# Creating a DiGraph
G = nx.DiGraph()

# Adding nodes and edges to the graph with color attributes for the top 10 rules
for i, row in final_rules.head(10).iterrows():
    for antecedent in row['antecedents']:
        G.add_node(antecedent, color='blue')
        G.add_edge(antecedent, f"R_{i + 1}", color='blue')
    for consequent in row['consequents']:
        G.add_node(consequent, color='orange')
        G.add_edge(f"R_{i + 1}", consequent, color='orange')

# Setting node colors based on 'color' attribute
node_colors = [G.nodes[node].get('color', 'red') for node in G.nodes]

# Setting edge colors based on 'color' attribute
edge_colors = [G.edges[edge].get('color', 'black') for edge in G.edges]

# Adjusting the layout algorithm
pos = nx.spring_layout(G)

# Drawing the graph
nx.draw(G, pos, with_labels=False, font_weight='bold', node_color=node_colors, edge_color=edge_colors)

# Drawing node labels with adjusted positions
node_labels = {node: node.split('_')[1] if 'R_' in node else node for node in G.nodes}
nx.draw_networkx_labels(G, pos, labels=node_labels, font_color='black', font_size=8)

# Displaying the graph
plt.show()



This visualisation aims to present the top 10 rules using a DiGraph. A DiGraph is a collection of nodes (vertices) and edges that connect pairs of nodes.
Each edge has a direction in a directed graph, indicating a one-way connection between two nodes. The blue nodes represent the antecedents and orange nodes represent the consequents. Edges represent the relationships or associations between antecedents and consequents in each rule.
Blue edges connect the antecedents to a rule node (e.g., "R_1," "R_2"), indicating the transition from antecedents to the association rule.
Orange edges connect the rule node to the consequents, indicating the transition from the association rule to the consequents.

Include description for Parallel coordinates plot

import plotly.express as px

# Convert rules to coordinates for a parallel coordinates plot
rules['antecedent'] = rules['antecedents'].apply(lambda antecedent: list(antecedent)[0])
rules['consequent'] = rules['consequents'].apply(lambda consequent: list(consequent)[0])
rules['rule'] = rules.index
# Define coordinates and label
coords = rules[['antecedent','consequent','rule']]

top_30_coords = coords.head(30)
fig = px.scatter(top_20_coords, x="antecedent", y="consequent", color="rule",
                 color_continuous_scale=px.colors.sequential.Viridis)

fig.show()


The directed plot seemed a little messy even for displaying top 10 rules. Hence we used this scatter plot to represent the top 20 rules. This scatter plot is used to visualise the relationships between the "antecedent" and "consequent" variables, and the color of each point is determined by the "rule" variable.

import pandas as pd
import plotly.graph_objects as go

top_20_rules = final_rules.head(20).copy()

# Convert frozenset to string in both antecedents and consequents to only display Investor names
top_20_rules['antecedents'] = top_20_rules['antecedents'].apply(lambda x: ', '.join(map(str, x)))
top_20_rules['consequents'] = top_20_rules['consequents'].apply(lambda x: ', '.join(map(str, x)))

# Create a DataFrame for Sankey diagram
sankey_df = pd.DataFrame(columns=['source', 'target', 'value'])

for index, row in top_20_rules.iterrows():
    antecedents_list = str(row['antecedents']).split(', ')
    consequents_list = str(row['consequents']).split(', ')

    for antecedent in antecedents_list:
        for consequent in consequents_list:
            # Ensure the source and target are not the same
            if antecedent != consequent:
                sankey_df = pd.concat([sankey_df, pd.DataFrame({'source': [antecedent], 'target': [consequent], 'value': [row['lift']]})])

# Create Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color='black', width=0.5),
        label=sankey_df['source'].append(sankey_df['target']).unique()
    ),
    link=dict(
        source=sankey_df['source'].map(lambda x: list(sankey_df['source'].unique()).index(x)),
        target=sankey_df['target'].map(lambda x: list(sankey_df['target'].unique()).index(x)),
        value=sankey_df['value']
    )
)])

# Update layout for better display
fig.update_layout(title_text="Sankey Diagram of Association Rules with Lift Values",
                  font_size=10)

# Show the plot
fig.show()


The Sankey diagram generated by this code aims to visually represent and analyze the association rules with lift values in a clear and intuitive manner. The thickness of the nodes (both antecedents and consequents) corresponds to the number of association rules associated with each investor or item. Thicker nodes indicate stronger associations. The lift values associated with each link (arrow) indicate the strength of the association between the antecedent and consequent. Higher lift values suggest stronger relationships.

## 2.2) Cluster Analysis on Companies

uni_small = data.drop(columns=['City', 'Year', 'Select Investors',
	'Deal Terms',	'Investors Count'])
uni_small.head()

# rounding values for Raised Total
uni_small["Raised Total ($B)"] = uni_small["Raised Total ($B)"].round(5)
uni_small.head()

import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


uni_small = uni_small.dropna()

# If you want to remove null values only for specific columns, you can use the subset parameter:
# uni_small = uni_small.dropna(subset=["GII Score", "GPDpercapita 2022", "Total Patent 10-22"])


#creating a new dataframe called company_counts to count the number of unicorns in each country

company_counts = uni_small['Country'].value_counts().reset_index()

# Rename the columns for clarity
company_counts.columns = ['Country', 'NumberofUnicorns']

company_counts_df = pd.DataFrame(company_counts)
company_counts_df.head()

merged_df = pd.merge(company_counts_df, uni_small, on='Country', how='inner')

# Select specific columns from uni_small
selected_columns = ['Country', 'NumberofUnicorns','GII Score', 'GPDpercapita 2022', 'Total Patent 10-22']

# Display the result with selected columns
result_df = merged_df.groupby('Country')[selected_columns].agg('mean').reset_index()

# Display the result
result_df.head()

#These 3 countries had NA values for GII/GDP per capita/Total Patent, hence we have removed them for clustering
countries_to_remove = ['Bermuda', 'Hong Kong', 'Bahamas']

# Remove rows for the specified countries
result_df = result_df[~result_df['Country'].isin(countries_to_remove)]
result_df



kmeans_df = result_df.drop('Country', axis=1)

# Display the updated result
print(kmeans_df)

# pair plot to view the relationship between the numeric variables
sns.pairplot(kmeans_df)

#scaling the values for k-means
ss=StandardScaler().fit_transform(kmeans_df)
df_data_std = pd.DataFrame(ss)
df_data_std

round(df_data_std.describe(),2)

# visualise - let's scale the axes to be consistent
sns.pairplot(df_data_std).set(xlim=(-5, 5), ylim=(-5, 5))



#we have created a loop from 2 to 10 to view the silhouette scores
from sklearn.metrics import silhouette_score
fits_2 = [0,0]
scores_2 = [0,0]
## For each iteration, a k-means clustering model is created with the specified number of clusters
for m in range(2,11):
  model = KMeans(n_clusters = m, n_init='auto').fit(kmeans_df)
  fits_2.append(model)
  scores_2.append(silhouette_score(kmeans_df, model.labels_, metric='euclidean'))

# to plot the silhouette score
sns.lineplot(x = range(0,11), y= scores_2)

The resulting plot should show how the silhouette score changes as the number of clusters in the k-means algorithm varies, helping to identify the optimal number of clusters for the given dataset. The optimal number is 3 clusters for this dataset.

kmeans = KMeans(n_clusters = 3, n_init='auto')
kmeans.fit(df_data_std)

pd.Series(kmeans.labels_).value_counts()

result_df['cluster_3']=kmeans.labels_

kmeans_df.to_csv("kmeans.csv")

from plotly.express import scatter_3d

result_df['cluster_3'] = result_df['cluster_3'].astype(str)

fig = scatter_3d(result_df,
                 x='Total Patent 10-22',
                 y='GII Score',
                 z='GPDpercapita 2022',
                 color='cluster_3',
                 opacity=0.5,
                 size='NumberofUnicorns',
                 size_max=110,  # Adjust the size_max parameter as needed
                 hover_data=['Country']  # Include 'Country' in hover information
                )

fig.show()


This code is creating a 3D scatter plot to visualize the relationship between three dimensions: 'Total Patent 10-22', 'GII Score', and 'GDP percapita 2022'. The data points are colored based on the 'cluster_3' column, and the size of the points is determined by the 'NumberofUnicorns' column.

result_df.to_csv("clustering_final.csv")
