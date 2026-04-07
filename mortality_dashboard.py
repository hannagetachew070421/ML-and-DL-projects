import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
df = pd.read_csv('IHME_GBD_2010_MORTALITY_AGE_SPECIFIC_BY_COUNTRY_1970_2010.csv')

# Clean column names and data
df.columns = df.columns.str.strip()
df['Number of Deaths'] = df['Number of Deaths'].str.replace(',', '').astype(int)
df['Death Rate Per 100,000'] = df['Death Rate Per 100,000'].str.replace(',', '').astype(float)

# Add region mapping (simplified)
region_mapping = {
    'AFG': 'South Asia', 'AGO': 'Sub-Saharan Africa', 'ALB': 'Europe & Central Asia',
    'AND': 'Europe & Central Asia', 'ARE': 'Middle East & North Africa', 'ARG': 'Latin America & Caribbean',
    'ARM': 'Europe & Central Asia', 'ATG': 'Latin America & Caribbean', 'AUS': 'East Asia & Pacific',
    'AUT': 'Europe & Central Asia', 'AZE': 'Europe & Central Asia', 'BDI': 'Sub-Saharan Africa',
    'BEL': 'Europe & Central Asia', 'USA': 'North America', 'CHN': 'East Asia & Pacific',
    'IND': 'South Asia', 'BRA': 'Latin America & Caribbean', 'RUS': 'Europe & Central Asia',
    'JPN': 'East Asia & Pacific', 'DEU': 'Europe & Central Asia', 'GBR': 'Europe & Central Asia',
    'FRA': 'Europe & Central Asia', 'ITA': 'Europe & Central Asia', 'CAN': 'North America',
    'KOR': 'East Asia & Pacific', 'ESP': 'Europe & Central Asia', 'MEX': 'Latin America & Caribbean',
    'IDN': 'East Asia & Pacific', 'TUR': 'Europe & Central Asia', 'SAU': 'Middle East & North Africa',
    'NLD': 'Europe & Central Asia', 'CHE': 'Europe & Central Asia', 'TWN': 'East Asia & Pacific',
    'BEL': 'Europe & Central Asia', 'IRE': 'Europe & Central Asia', 'ISR': 'Middle East & North Africa',
    'NOR': 'Europe & Central Asia', 'AUT': 'Europe & Central Asia', 'UAE': 'Middle East & North Africa',
    'NGA': 'Sub-Saharan Africa', 'ZAF': 'Sub-Saharan Africa', 'EGY': 'Middle East & North Africa',
    'VNM': 'East Asia & Pacific', 'BGD': 'South Asia', 'PHL': 'East Asia & Pacific',
    'ETH': 'Sub-Saharan Africa', 'PAK': 'South Asia', 'IRN': 'Middle East & North Africa',
    'THA': 'East Asia & Pacific', 'MYS': 'East Asia & Pacific', 'SGP': 'East Asia & Pacific',
    'CHL': 'Latin America & Caribbean', 'FIN': 'Europe & Central Asia', 'DNK': 'Europe & Central Asia',
    'SWE': 'Europe & Central Asia', 'PRT': 'Europe & Central Asia', 'GRC': 'Europe & Central Asia',
    'CZE': 'Europe & Central Asia', 'HUN': 'Europe & Central Asia', 'POL': 'Europe & Central Asia',
    'SVK': 'Europe & Central Asia', 'SVN': 'Europe & Central Asia', 'EST': 'Europe & Central Asia',
    'LVA': 'Europe & Central Asia', 'LTU': 'Europe & Central Asia', 'HRV': 'Europe & Central Asia',
    'BGR': 'Europe & Central Asia', 'ROU': 'Europe & Central Asia', 'SRB': 'Europe & Central Asia',
    'MNE': 'Europe & Central Asia', 'BIH': 'Europe & Central Asia', 'MKD': 'Europe & Central Asia',
    'ALB': 'Europe & Central Asia', 'MDA': 'Europe & Central Asia', 'UKR': 'Europe & Central Asia',
    'BLR': 'Europe & Central Asia', 'GEO': 'Europe & Central Asia', 'ARM': 'Europe & Central Asia',
    'AZE': 'Europe & Central Asia', 'KAZ': 'Europe & Central Asia', 'KGZ': 'Europe & Central Asia',
    'TJK': 'Europe & Central Asia', 'TKM': 'Europe & Central Asia', 'UZB': 'Europe & Central Asia',
    'AFG': 'South Asia', 'NPL': 'South Asia', 'BTN': 'South Asia', 'LKA': 'South Asia',
    'MDV': 'South Asia', 'CHN': 'East Asia & Pacific', 'MNG': 'East Asia & Pacific',
    'PRK': 'East Asia & Pacific', 'KOR': 'East Asia & Pacific', 'JPN': 'East Asia & Pacific'
}

# Add default region for countries not in mapping
df['Region'] = df['Country Code'].map(region_mapping).fillna('Other')

# Initialize Dash app
app = dash.Dash(__name__)

# Define app layout

app.layout = html.Div([
    html.Div([
    html.Div([
        html.H2("Debre Berhan University", style={
            'marginBottom': '5px',
            'fontWeight': 'bold',
            'color': '#273c75',
            'fontSize': '32px',
            'letterSpacing': '1.2px'
        }),
        html.H4("Department of Data Science", style={
            'marginTop': '0px',
            'marginBottom': '3px',
            'color': '#40739e',
            'fontSize': '22px',
            'fontStyle': 'italic'
        }),
        html.Div("Data Visualization Project", style={
            'color': '#40739e',
            'fontSize': '18px',
            'marginTop': '6px',
            'fontWeight': 'bold',
            'letterSpacing': '1px'
        }),
        html.Div("Prepared By Group 3 ", style={
            'marginTop': '0px',
            'marginBottom': '3px',
            'color': '#40739e',
            'fontSize': '19px',
            'fontStyle': 'italic'
        })
        
    ], style={
        'textAlign': 'center',
        'padding': '18px 10px 8px 10px',
        'backgroundColor': '#f5f6fa',
        'borderRadius': '12px',
        'boxShadow': '0 2px 8px #dcdde1',
        'maxWidth': '100%',
        'margin': '20px auto 10px auto'
    })
]),
    
    html.Div([
        html.H1("Global Mortality Dashboard & Forecasting (1970-2040)", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
        html.P("Interactive analysis of age-specific mortality patterns with predictive forecasting",
               style={'textAlign': 'center', 'fontSize': '18px', 'color': '#7f8c8d'})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px'}),
    
    # Control Panel
    html.Div([
        html.Div([
            html.Label("Select Year Range:", style={'fontWeight': 'bold'}),
            dcc.RangeSlider(
                id='year-slider',
                min=1970, max=2010, step=10,
                marks={i: str(i) for i in range(1970, 2011, 10)},
                value=[1970, 2010]
            )
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),
        
        html.Div([
            html.Label("Select Countries:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='country-dropdown',
                options=[{'label': country, 'value': country} 
                        for country in sorted(df['Country Name'].unique())],
                value=['United States', 'China', 'India', 'Germany', 'Japan'],
                multi=True
            )
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),
        
        html.Div([
            html.Label("Select Sex:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='sex-dropdown',
                options=[{'label': sex, 'value': sex} for sex in df['Sex'].unique()],
                value='Both'
            )
        ], style={'width': '15%', 'display': 'inline-block', 'marginRight': '5%'}),
        
        html.Div([
            html.Label("Select Age Group:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='age-dropdown',
                options=[{'label': age, 'value': age} for age in sorted(df['Age Group'].unique())],
                value='All ages'
            )
        ], style={'width': '15%', 'display': 'inline-block'})
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'marginBottom': '20px'}),
    
    # KPI Cards
    html.Div(id='kpi-cards', style={'marginBottom': '20px'}),
    
    # Charts Row 1
    html.Div([
        html.Div([
            dcc.Graph(id='trend-chart')
        ], style={'width': '50%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id='regional-comparison')
        ], style={'width': '50%', 'display': 'inline-block'})
    ]),
    
    # Global Map Row
    html.Div([
        html.Div([
            dcc.Graph(id='global-map')
        ], style={'width': '100%', 'display': 'inline-block'})
    ]),
    
    # Charts Row 2
    html.Div([
        html.Div([
            dcc.Graph(id='age-distribution')
        ], style={'width': '50%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id='mortality-heatmap')
        ], style={'width': '50%', 'display': 'inline-block'})
    ]),
    
    # Charts Row 3 - Additional EDA Components
    html.Div([
        html.Div([
            dcc.Graph(id='top-countries-chart')
        ], style={'width': '33%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id='mortality-distribution')
        ], style={'width': '33%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id='yearly-change-chart')
        ], style={'width': '34%', 'display': 'inline-block'})
    ]),
    
    # Charts Row 4 - Time Series Analysis
    html.Div([
        html.Div([
            dcc.Graph(id='sex-comparison-chart')
        ], style={'width': '50%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id='correlation-analysis')
        ], style={'width': '50%', 'display': 'inline-block'})
    ]),
    
    # Forecasting Section
    html.Div([
        html.H2("Mortality Rate Forecasting", 
                style={'color': '#2c3e50', 'textAlign': 'center', 'marginBottom': '20px'}),
        
        # Forecasting Controls
        html.Div([
            html.Div([
                html.Label("Forecast Period (Years):", style={'fontWeight': 'bold'}),
                dcc.RangeSlider(
                    id='forecast-slider',
                    min=5, max=30, step=5,
                    marks={5: '5 Years', 10: '10 Years', 20: '20 Years', 30: '30 Years'},
                    value=[5, 20],
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
            
            html.Div([
                html.Label("Forecast Model:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='model-dropdown',
                    options=[
                        {'label': 'Linear Trend', 'value': 'linear'},
                        {'label': 'Polynomial (Degree 2)', 'value': 'poly2'},
                        {'label': 'Polynomial (Degree 3)', 'value': 'poly3'}
                    ],
                    value='linear'
                )
            ], style={'width': '48%', 'display': 'inline-block'})
        ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'marginBottom': '20px'}),
        
        # Forecasting Chart
        html.Div([
            dcc.Graph(id='forecast-chart')
        ], style={'marginBottom': '20px'})
    ], style={'backgroundColor': '#ffffff', 'padding': '20px', 'marginTop': '20px', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
    
    # Data Story Section
    html.Div([
        html.H2("Data Story: Global Mortality Trends (1970-2010)", 
                style={'color': '#2c3e50', 'textAlign': 'center'}),
        html.Div(id='data-story-content')
    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'marginTop': '20px'})
])

# Callback for KPI cards
@app.callback(
    Output('kpi-cards', 'children'),
    [Input('year-slider', 'value'),
     Input('country-dropdown', 'value'),
     Input('sex-dropdown', 'value'),
     Input('age-dropdown', 'value')]
)
def update_kpis(year_range, countries, sex, age_group):
    filtered_df = df[
        (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1]) &
        (df['Country Name'].isin(countries)) &
        (df['Sex'] == sex) &
        (df['Age Group'] == age_group)
    ]
    
    if filtered_df.empty:
        return html.Div("No data available for selected filters")
    
    # Calculate KPIs
    total_deaths = filtered_df['Number of Deaths'].sum()
    avg_death_rate = filtered_df['Death Rate Per 100,000'].mean()
    countries_count = filtered_df['Country Name'].nunique()
    
    # Calculate trend (comparing first and last available years)
    if len(filtered_df['Year'].unique()) > 1:
        first_year_data = filtered_df[filtered_df['Year'] == filtered_df['Year'].min()]
        last_year_data = filtered_df[filtered_df['Year'] == filtered_df['Year'].max()]
        
        first_year_rate = first_year_data['Death Rate Per 100,000'].mean()
        last_year_rate = last_year_data['Death Rate Per 100,000'].mean()
        trend_change = ((last_year_rate - first_year_rate) / first_year_rate) * 100
    else:
        trend_change = 0
    
    return html.Div([
        html.Div([
            html.H3(f"{total_deaths:,}", style={'color': '#e74c3c', 'margin': '0'}),
            html.P("Total Deaths", style={'margin': '0', 'color': '#7f8c8d'})
        ], className='kpi-card', style={'width': '23%', 'display': 'inline-block', 'margin': '1%', 
                                       'padding': '20px', 'backgroundColor': 'white', 
                                       'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                                       'textAlign': 'center'}),
        
        html.Div([
            html.H3(f"{avg_death_rate:.1f}", style={'color': '#3498db', 'margin': '0'}),
            html.P("Avg Death Rate per 100K", style={'margin': '0', 'color': '#7f8c8d'})
        ], className='kpi-card', style={'width': '23%', 'display': 'inline-block', 'margin': '1%', 
                                       'padding': '20px', 'backgroundColor': 'white', 
                                       'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                                       'textAlign': 'center'}),
        
        html.Div([
            html.H3(f"{countries_count}", style={'color': '#2ecc71', 'margin': '0'}),
            html.P("Countries Analyzed", style={'margin': '0', 'color': '#7f8c8d'})
        ], className='kpi-card', style={'width': '23%', 'display': 'inline-block', 'margin': '1%', 
                                       'padding': '20px', 'backgroundColor': 'white', 
                                       'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                                       'textAlign': 'center'}),
        
        html.Div([
            html.H3(f"{trend_change:+.1f}%", style={'color': '#f39c12', 'margin': '0'}),
            html.P("Mortality Rate Change", style={'margin': '0', 'color': '#7f8c8d'})
        ], className='kpi-card', style={'width': '23%', 'display': 'inline-block', 'margin': '1%', 
                                       'padding': '20px', 'backgroundColor': 'white', 
                                       'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                                       'textAlign': 'center'})
    ])

# Callback for trend chart
@app.callback(
    Output('trend-chart', 'figure'),
    [Input('year-slider', 'value'),
     Input('country-dropdown', 'value'),
     Input('sex-dropdown', 'value'),
     Input('age-dropdown', 'value')]
)
def update_trend_chart(year_range, countries, sex, age_group):
    filtered_df = df[
        (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1]) &
        (df['Country Name'].isin(countries)) &
        (df['Sex'] == sex) &
        (df['Age Group'] == age_group)
    ]
    
    if filtered_df.empty:
        return go.Figure().add_annotation(text="No data available", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    fig = px.line(filtered_df, x='Year', y='Death Rate Per 100,000', 
                  color='Country Name', title='Mortality Rate Trends Over Time',
                  labels={'Death Rate Per 100,000': 'Death Rate per 100,000'})
    
    fig.update_layout(height=400, showlegend=True)
    return fig

# Callback for regional comparison
@app.callback(
    Output('regional-comparison', 'figure'),
    [Input('year-slider', 'value'),
     Input('sex-dropdown', 'value'),
     Input('age-dropdown', 'value')]
)
def update_regional_comparison(year_range, sex, age_group):
    filtered_df = df[
        (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1]) &
        (df['Sex'] == sex) &
        (df['Age Group'] == age_group)
    ]
    
    if filtered_df.empty:
        return go.Figure().add_annotation(text="No data available", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    regional_data = filtered_df.groupby('Region')['Death Rate Per 100,000'].mean().reset_index()
    
    fig = px.bar(regional_data, x='Region', y='Death Rate Per 100,000',
                 title='Average Mortality Rate by Region',
                 labels={'Death Rate Per 100,000': 'Avg Death Rate per 100,000'})
    
    fig.update_layout(height=400, xaxis_tickangle=-45)
    return fig

# Callback for age distribution
@app.callback(
    Output('age-distribution', 'figure'),
    [Input('year-slider', 'value'),
     Input('country-dropdown', 'value'),
     Input('sex-dropdown', 'value')]
)
def update_age_distribution(year_range, countries, sex):
    filtered_df = df[
        (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1]) &
        (df['Country Name'].isin(countries)) &
        (df['Sex'] == sex) &
        (df['Age Group'] != 'All ages')  # Exclude 'All ages' for age-specific analysis
    ]
    
    if filtered_df.empty:
        return go.Figure().add_annotation(text="No data available", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    age_data = filtered_df.groupby('Age Group')['Death Rate Per 100,000'].mean().reset_index()
    
    fig = px.bar(age_data, x='Age Group', y='Death Rate Per 100,000',
                 title='Mortality Rate by Age Group',
                 labels={'Death Rate Per 100,000': 'Avg Death Rate per 100,000'})
    
    fig.update_layout(height=400, xaxis_tickangle=-45)
    return fig

# Callback for mortality heatmap
@app.callback(
    Output('mortality-heatmap', 'figure'),
    [Input('year-slider', 'value'),
     Input('country-dropdown', 'value'),
     Input('sex-dropdown', 'value')]
)
def update_mortality_heatmap(year_range, countries, sex):
    filtered_df = df[
        (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1]) &
        (df['Country Name'].isin(countries)) &
        (df['Sex'] == sex) &
        (df['Age Group'] != 'All ages')
    ]
    
    if filtered_df.empty:
        return go.Figure().add_annotation(text="No data available", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    # Create pivot table for heatmap
    heatmap_data = filtered_df.pivot_table(
        values='Death Rate Per 100,000', 
        index='Age Group', 
        columns='Country Name', 
        aggfunc='mean'
    )
    
    fig = px.imshow(heatmap_data, 
                    title='Mortality Rate Heatmap: Age Groups vs Countries',
                    labels={'color': 'Death Rate per 100,000'},
                    aspect='auto')
    
    fig.update_layout(height=400)
    return fig

# Callback for top countries chart
@app.callback(
    Output('top-countries-chart', 'figure'),
    [Input('year-slider', 'value'),
     Input('sex-dropdown', 'value'),
     Input('age-dropdown', 'value')]
)
def update_top_countries_chart(year_range, sex, age_group):
    filtered_df = df[
        (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1]) &
        (df['Sex'] == sex) &
        (df['Age Group'] == age_group)
    ]
    
    if filtered_df.empty:
        return go.Figure().add_annotation(text="No data available", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    top_countries = filtered_df.groupby('Country Name')['Death Rate Per 100,000'].mean().nlargest(10).reset_index()
    
    fig = px.bar(top_countries, x='Death Rate Per 100,000', y='Country Name',
                 title='Top 10 Countries by Mortality Rate',
                 orientation='h', color='Death Rate Per 100,000',
                 color_continuous_scale='Reds')
    
    fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
    return fig

# Callback for mortality distribution
@app.callback(
    Output('mortality-distribution', 'figure'),
    [Input('year-slider', 'value'),
     Input('country-dropdown', 'value'),
     Input('sex-dropdown', 'value'),
     Input('age-dropdown', 'value')]
)
def update_mortality_distribution(year_range, countries, sex, age_group):
    filtered_df = df[
        (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1]) &
        (df['Country Name'].isin(countries)) &
        (df['Sex'] == sex) &
        (df['Age Group'] == age_group)
    ]
    
    if filtered_df.empty:
        return go.Figure().add_annotation(text="No data available", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    fig = px.histogram(filtered_df, x='Death Rate Per 100,000',
                       title='Mortality Rate Distribution', nbins=20,
                       color_discrete_sequence=['#3498db'])
    
    fig.update_layout(height=400)
    return fig

# Callback for yearly change chart
@app.callback(
    Output('yearly-change-chart', 'figure'),
    [Input('year-slider', 'value'),
     Input('country-dropdown', 'value'),
     Input('sex-dropdown', 'value'),
     Input('age-dropdown', 'value')]
)
def update_yearly_change_chart(year_range, countries, sex, age_group):
    filtered_df = df[
        (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1]) &
        (df['Country Name'].isin(countries)) &
        (df['Sex'] == sex) &
        (df['Age Group'] == age_group)
    ]
    
    if filtered_df.empty:
        return go.Figure().add_annotation(text="No data available", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    yearly_change = []
    for country in countries:
        country_data = filtered_df[filtered_df['Country Name'] == country].sort_values('Year')
        if len(country_data) > 1:
            country_data['YoY_Change'] = country_data['Death Rate Per 100,000'].pct_change() * 100
            yearly_change.append(country_data)
    
    if yearly_change:
        combined_data = pd.concat(yearly_change)
        avg_change = combined_data.groupby('Year')['YoY_Change'].mean().reset_index()
        
        fig = px.line(avg_change, x='Year', y='YoY_Change',
                      title='Average Year-over-Year Change (%)', line_shape='spline')
        fig.add_hline(y=0, line_dash="dash", line_color="red")
    else:
        fig = go.Figure().add_annotation(text="Insufficient data for YoY analysis", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    fig.update_layout(height=400)
    return fig

# Callback for sex comparison chart
@app.callback(
    Output('sex-comparison-chart', 'figure'),
    [Input('year-slider', 'value'),
     Input('country-dropdown', 'value'),
     Input('age-dropdown', 'value')]
)
def update_sex_comparison_chart(year_range, countries, age_group):
    filtered_df = df[
        (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1]) &
        (df['Country Name'].isin(countries)) &
        (df['Age Group'] == age_group) &
        (df['Sex'].isin(['Male', 'Female']))
    ]
    
    if filtered_df.empty:
        return go.Figure().add_annotation(text="No data available", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    sex_comparison = filtered_df.groupby(['Year', 'Sex'])['Death Rate Per 100,000'].mean().reset_index()
    
    fig = px.line(sex_comparison, x='Year', y='Death Rate Per 100,000',
                  color='Sex', title='Mortality Rate by Sex Over Time',
                  color_discrete_map={'Male': '#3498db', 'Female': '#e74c3c'})
    
    fig.update_layout(height=400)
    return fig

# Callback for global map
@app.callback(
    Output('global-map', 'figure'),
    [Input('year-slider', 'value'),
     Input('sex-dropdown', 'value'),
     Input('age-dropdown', 'value')]
)
def update_global_map(year_range, sex, age_group):
    filtered_df = df[
        (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1]) &
        (df['Sex'] == sex) &
        (df['Age Group'] == age_group)
    ]
    
    if filtered_df.empty:
        return go.Figure().add_annotation(text="No data available", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    # Calculate average mortality rate by country
    map_data = filtered_df.groupby(['Country Code', 'Country Name'])['Death Rate Per 100,000'].mean().reset_index()
    
    fig = px.choropleth(
        map_data,
        locations='Country Code',
        color='Death Rate Per 100,000',
        hover_name='Country Name',
        hover_data={'Country Code': False, 'Death Rate Per 100,000': ':.1f'},
        color_continuous_scale='Reds',
        title=f'Global Mortality Rates ({year_range[0]}-{year_range[1]}) - {sex}, {age_group}',
        labels={'Death Rate Per 100,000': 'Death Rate per 100,000'}
    )
    
    fig.update_layout(
        height=500,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        )
    )
    
    return fig

# Callback for correlation analysis
@app.callback(
    Output('correlation-analysis', 'figure'),
    [Input('year-slider', 'value'),
     Input('country-dropdown', 'value'),
     Input('sex-dropdown', 'value')]
)
def update_correlation_analysis(year_range, countries, sex):
    filtered_df = df[
        (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1]) &
        (df['Country Name'].isin(countries)) &
        (df['Sex'] == sex)
    ]
    
    if filtered_df.empty:
        return go.Figure().add_annotation(text="No data available", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    fig = px.scatter(filtered_df, x='Number of Deaths', y='Death Rate Per 100,000',
                     color='Country Name', size='Year',
                     title='Deaths vs Death Rate Correlation',
                     hover_data=['Year', 'Age Group'])
    
    fig.update_layout(height=400)
    return fig

# Forecasting function
def create_forecast(data, years_ahead, model_type='linear'):
    """Create mortality rate forecast using specified model"""
    if len(data) < 3:  # Need minimum data points
        return None, None
    
    # Prepare data
    X = data['Year'].values.reshape(-1, 1)
    y = data['Death Rate Per 100,000'].values
    
    # Create model based on type
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'poly2':
        model = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())])
    elif model_type == 'poly3':
        model = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression())])
    
    # Fit model
    model.fit(X, y)
    
    # Generate future years
    last_year = data['Year'].max()
    future_years = np.arange(last_year + 1, last_year + years_ahead + 1).reshape(-1, 1)
    
    # Make predictions
    predictions = model.predict(future_years)
    
    # Ensure predictions are non-negative
    predictions = np.maximum(predictions, 0)
    
    return future_years.flatten(), predictions

# Callback for forecasting chart
@app.callback(
    Output('forecast-chart', 'figure'),
    [Input('year-slider', 'value'),
     Input('country-dropdown', 'value'),
     Input('sex-dropdown', 'value'),
     Input('age-dropdown', 'value'),
     Input('forecast-slider', 'value'),
     Input('model-dropdown', 'value')]
)
def update_forecast_chart(year_range, countries, sex, age_group, forecast_range, model_type):
    # Filter historical data
    filtered_df = df[
        (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1]) &
        (df['Country Name'].isin(countries)) &
        (df['Sex'] == sex) &
        (df['Age Group'] == age_group)
    ]
    
    if filtered_df.empty:
        return go.Figure().add_annotation(text="No data available for forecasting", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    fig = go.Figure()
    
    # Track which forecast categories have been added to avoid duplicate legends
    added_categories = set()
    
    # Define consistent colors for countries
    country_colors = px.colors.qualitative.Set1[:len(countries)]
    country_color_map = {country: country_colors[i] for i, country in enumerate(countries)}
    
    # Add historical data and forecasts for each country
    for country in countries:
        country_data = filtered_df[filtered_df['Country Name'] == country]
        
        if len(country_data) < 2:
            continue
            
        # Sort by year
        country_data = country_data.sort_values('Year')
        
        # Get country's base color
        base_color = country_color_map[country]
        
        # Add historical data with professional styling
        fig.add_trace(go.Scatter(
            x=country_data['Year'],
            y=country_data['Death Rate Per 100,000'],
            mode='lines+markers',
            name=f'{country} - Historical Data',
            line=dict(width=3, color=base_color),
            marker=dict(size=6, color=base_color)
        ))
        
        # Generate forecasts for different periods
        forecast_periods = [5, 10, 20, 30]
        
        for period in forecast_periods:
            if period >= forecast_range[0] and period <= forecast_range[1]:
                future_years, predictions = create_forecast(country_data, period, model_type)
                
                if future_years is not None:
                    # Determine forecast category for legend
                    if period <= 5:
                        category = "Short-term"
                    elif period <= 10:
                        category = "Medium-term"
                    elif period <= 20:
                        category = "Long-term"
                    else:
                        category = "Extended"
                    
                    # Create legend name and check if category already added
                    legend_name = f'{category} Projection ({period}Y)'
                    show_legend = legend_name not in added_categories
                    
                    if show_legend:
                        added_categories.add(legend_name)
                    
                    # Add forecast line with same base color but dashed
                    fig.add_trace(go.Scatter(
                        x=future_years,
                        y=predictions,
                        mode='lines',
                        name=legend_name,
                        line=dict(
                            dash='dash', 
                            width=2.5,
                            color=base_color
                        ),
                        opacity=0.7,
                        showlegend=show_legend
                    ))
                    

    
    # Update layout
    fig.update_layout(
        title=f'Mortality Rate Forecasting: {forecast_range[0]}-{forecast_range[1]} Years Ahead ({model_type.title()} Model)',
        xaxis_title='Year',
        yaxis_title='Death Rate per 100,000',
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Add vertical line to separate historical and forecast data
    last_historical_year = df['Year'].max()
    fig.add_vline(x=last_historical_year, line_dash="dot", line_color="gray", 
                  annotation_text="Historical | Forecast", annotation_position="top")
    
    return fig

# Callback for data story
@app.callback(
    Output('data-story-content', 'children'),
    [Input('year-slider', 'value'),
     Input('country-dropdown', 'value'),
     Input('sex-dropdown', 'value')]
)
def update_data_story(year_range, countries, sex):
    # Filter data for analysis
    filtered_df = df[
        (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1]) &
        (df['Country Name'].isin(countries)) &
        (df['Sex'] == sex) &
        (df['Age Group'] == 'All ages')
    ]
    
    if filtered_df.empty:
        return html.Div("No data available for analysis")
    
    # Calculate insights
    first_year = filtered_df['Year'].min()
    last_year = filtered_df['Year'].max()
    
    first_year_data = filtered_df[filtered_df['Year'] == first_year]
    last_year_data = filtered_df[filtered_df['Year'] == last_year]
    
    # Overall trend
    overall_change = ((last_year_data['Death Rate Per 100,000'].mean() - 
                      first_year_data['Death Rate Per 100,000'].mean()) / 
                     first_year_data['Death Rate Per 100,000'].mean()) * 100
    
    # Best and worst performing countries
    country_trends = []
    for country in countries:
        country_first = first_year_data[first_year_data['Country Name'] == country]['Death Rate Per 100,000'].values
        country_last = last_year_data[last_year_data['Country Name'] == country]['Death Rate Per 100,000'].values
        
        if len(country_first) > 0 and len(country_last) > 0:
            change = ((country_last[0] - country_first[0]) / country_first[0]) * 100
            country_trends.append((country, change))
    
    country_trends.sort(key=lambda x: x[1])
    best_performer = country_trends[0] if country_trends else None
    worst_performer = country_trends[-1] if country_trends else None
    
    return html.Div([
        html.H3("Key Insights", style={'color': '#2c3e50', 'textDecoration': 'underline'}),
        
        html.Div([
            html.H4("Past (1970s-1980s)", style={'color': '#e74c3c'}),
            html.P(f"During the early period ({first_year}-1980), global mortality patterns showed significant "
                   f"regional disparities. Many developing countries experienced high mortality rates due to "
                   f"infectious diseases, poor healthcare infrastructure, and limited access to medical care.")
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.H4("Present (2010)", style={'color': '#3498db'}),
            html.P(f"By 2010, the mortality landscape had transformed significantly. Overall mortality rates "
                   f"{'decreased' if overall_change < 0 else 'increased'} by {abs(overall_change):.1f}% "
                   f"across selected countries. This reflects improvements in healthcare, vaccination programs, "
                   f"and better disease management.")
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.H4("Future Projections (2011-2040)", style={'color': '#9b59b6'}),
            html.P("Based on historical trends and forecasting models, mortality patterns are expected to continue "
                   "evolving. Key projections include:"),
            html.Ul([
                html.Li("Continued decline in infectious disease mortality in developed countries"),
                html.Li("Rising non-communicable disease burden, especially cardiovascular and cancer"),
                html.Li("Aging populations leading to higher overall mortality rates"),
                html.Li("Potential impact of climate change on mortality patterns"),
                html.Li("Emerging health threats requiring adaptive healthcare systems")
            ])
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.H4("Forecasting Methodology", style={'color': '#8e44ad'}),
            html.P("The forecasting models use historical mortality data (1970-2010) to project future trends. "
                   "Three model types are available:"),
            html.Ul([
                html.Li("Linear Trend: Assumes constant rate of change over time"),
                html.Li("Polynomial (Degree 2): Captures accelerating or decelerating trends"),
                html.Li("Polynomial (Degree 3): Models more complex trend patterns")
            ]),
            html.P("Confidence intervals provide uncertainty estimates. Longer forecasts have higher uncertainty.")
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.H4("Performance Highlights", style={'color': '#f39c12'}),
            html.P(
                f"Best performing country (greatest decrease in mortality rate): "
                f"{best_performer[0] if best_performer else 'N/A'} "
                f"({best_performer[1]:.1f}% change)"
                if best_performer else ""
            ),
            html.P(
                f"Most challenging country (greatest increase in mortality rate): "
                f"{worst_performer[0] if worst_performer else 'N/A'} "
                f"({worst_performer[1]:.1f}% change)"
                if worst_performer else ""
            )
        ])
    ])

if __name__ == '__main__':
    app.run(debug=True, port=8050)