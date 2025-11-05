import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime
import re
import smtplib
from email.message import EmailMessage
import warnings
warnings.filterwarnings('ignore')

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.LITERA])

# Load and preprocess data
def load_data():
    """Load and clean employee dataset"""
    try:
        # Update this path to your actual CSV file location
        df = pd.read_csv('employee_dataset.csv')
        
        # Data cleaning
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.dropna(axis=1, how='all')
        
        # Parse hire date with format specification to avoid warnings
        df['Hire_Date'] = pd.to_datetime(df['Hire_Date'], format='%Y-%m-%d', errors='coerce')
        df = df.dropna(subset=['Hire_Date'])
        
        # Handle missing values
        for col in ['Performance_Score', 'Satisfaction_Score', 'Productivity']:
            if col in df.columns:
                df[col].fillna(df.groupby('Department')[col].transform('median'), inplace=True)
        
        # Extract date components
        df['Hire_Year'] = df['Hire_Date'].dt.year
        df['Hire_Month'] = df['Hire_Date'].dt.month
        df['Hire_Day'] = df['Hire_Date'].dt.day
        
        return df
    except FileNotFoundError:
        # Create sample data if file not found
        return create_sample_data()

def create_sample_data():
    """Create sample employee data for testing"""
    import numpy as np
    np.random.seed(42)
    
    n = 150
    departments = ['Engineering', 'Sales', 'HR', 'Operations', 'Marketing']
    job_titles = ['Manager', 'Senior Analyst', 'Analyst', 'Engineer', 'Specialist']
    
    df = pd.DataFrame({
        'Employee_ID': [f'EMP{i:04d}' for i in range(1, n+1)],
        'Name': [f'Employee {i}' for i in range(1, n+1)],
        'Department': np.random.choice(departments, n),
        'Job_Title': np.random.choice(job_titles, n),
        'Age': np.random.randint(25, 60, n),
        'Annual_Salary': np.random.randint(350000, 2800000, n),
        'Years_At_Company': np.random.randint(0, 15, n),
        'Performance_Score': np.random.uniform(4.0, 9.5, n).round(1),
        'Satisfaction_Score': np.random.uniform(3.0, 9.0, n).round(1),
        'Productivity': np.random.uniform(50, 95, n).round(1),
        'Projects_Handled': np.random.randint(1, 8, n),
        'Remote_Work_Frequency': np.random.choice(['On-site', 'Hybrid', 'Remote'], n),
        'Retention_Risk_Index': np.random.uniform(0.5, 2.5, n).round(2),
        'Hire_Date': pd.date_range(start='2015-01-01', periods=n, freq='15D')
    })
    
    df['Hire_Year'] = df['Hire_Date'].dt.year
    df['Hire_Month'] = df['Hire_Date'].dt.month
    df['Hire_Day'] = df['Hire_Date'].dt.day
    
    return df

# Load data
df = load_data()

# Synonym mapping for chatbot
SYNONYM_MAP = {
    'salary': 'Annual_Salary',
    'pay': 'Annual_Salary',
    'earnings': 'Annual_Salary',
    'compensation': 'Annual_Salary',
    'performance': 'Performance_Score',
    'rating': 'Performance_Score',
    'evaluation': 'Performance_Score',
    'satisfaction': 'Satisfaction_Score',
    'happiness': 'Satisfaction_Score',
    'morale': 'Satisfaction_Score',
    'productivity': 'Productivity',
    'output': 'Productivity',
    'tenure': 'Years_At_Company',
    'experience': 'Years_At_Company',
    'seniority': 'Years_At_Company',
    'projects': 'Projects_Handled',
    'age': 'Age'
}

# Predefined Q&A
PREDEFINED_QA = {
    'what departments exist': lambda: f"Departments: {', '.join(df['Department'].unique())}",
    'list all departments': lambda: f"Departments: {', '.join(df['Department'].unique())}",
    'show departments': lambda: f"Departments: {', '.join(df['Department'].unique())}",
    'what job titles exist': lambda: f"Job Titles: {', '.join(df['Job_Title'].unique())}",
    'list job titles': lambda: f"Job Titles: {', '.join(df['Job_Title'].unique())}",
    'total employees': lambda: f"Total Employees: {len(df)}",
    'how many employees': lambda: f"Total Employees: {len(df)}",
    'employee count': lambda: f"Total Employees: {len(df)}",
}

def process_chatbot_query(query, filtered_df):
    """Process natural language queries"""
    query_lower = query.lower().strip()
    
    # Check predefined answers
    for key, func in PREDEFINED_QA.items():
        if key in query_lower:
            return func()
    
    # Extract aggregation type
    agg_type = None
    if re.search(r'\b(average|mean|avg)\b', query_lower):
        agg_type = 'mean'
    elif re.search(r'\b(total|sum)\b', query_lower):
        agg_type = 'sum'
    elif re.search(r'\b(maximum|max|highest)\b', query_lower):
        agg_type = 'max'
    elif re.search(r'\b(minimum|min|lowest)\b', query_lower):
        agg_type = 'min'
    
    # Find metric in query
    metric_col = None
    for synonym, col in SYNONYM_MAP.items():
        if synonym in query_lower and col in filtered_df.columns:
            metric_col = col
            break
    
    # Group-wise analysis
    if ' by ' in query_lower and metric_col:
        if 'department' in query_lower:
            result = filtered_df.groupby('Department')[metric_col].mean().round(2)
            return f"Average {metric_col} by Department:\n" + "\n".join([f"{k}: {v}" for k, v in result.items()])
        elif 'job title' in query_lower or 'role' in query_lower:
            result = filtered_df.groupby('Job_Title')[metric_col].mean().round(2)
            return f"Average {metric_col} by Job Title:\n" + "\n".join([f"{k}: {v}" for k, v in result.items()])
    
    # Simple aggregation
    if metric_col and agg_type:
        value = filtered_df[metric_col].agg(agg_type)
        if 'Salary' in metric_col:
            return f"The {agg_type} {metric_col.replace('_', ' ')} is ₹{value:,.0f}"
        else:
            return f"The {agg_type} {metric_col.replace('_', ' ')} is {value:.2f}"
    
    # Top N queries
    if re.search(r'\b(top|best|highest)\s+(\d+)', query_lower):
        n = int(re.search(r'\b(top|best|highest)\s+(\d+)', query_lower).group(2))
        if metric_col:
            top_employees = filtered_df.nlargest(n, metric_col)[['Name', 'Department', metric_col]]
            result = f"Top {n} by {metric_col}:\n"
            for idx, row in top_employees.iterrows():
                result += f"{row['Name']} ({row['Department']}): {row[metric_col]}\n"
            return result
    
    return "I didn't understand that query. Try asking about average salary, performance by department, or top 5 performers."

def send_email_alert(high_risk_employees):
    """Send email alert for high-risk employees"""
    try:
        # Configure your email settings here
        sender_email = "abhishekgantana@gmail.com"
        receiver_email = "abhishekgantana@gmail.com"  # FIXED: Added @gmail.com
        password = "kccv wmse wsjq nucx"
        
        msg = EmailMessage()
        msg['Subject'] = f'Retention Alert - {len(high_risk_employees)} High-Risk Employees'
        msg['From'] = sender_email
        msg['To'] = receiver_email
        
        # Create HTML content
        html_content = f"""
        <html>
            <body>
                <h2>Employee Retention Alert</h2>
                <p>The following {len(high_risk_employees)} employees have been identified as high retention risk (Risk Index > 1.5):</p>
                <table border="1" style="border-collapse: collapse; width: 100%;">
                    <tr style="background-color: #f2f2f2;">
                        <th>Employee ID</th>
                        <th>Name</th>
                        <th>Department</th>
                        <th>Satisfaction</th>
                        <th>Performance</th>
                        <th>Risk Index</th>
                    </tr>
        """
        
        for _, row in high_risk_employees.iterrows():
            html_content += f"""
                    <tr>
                        <td>{row['Employee_ID']}</td>
                        <td>{row['Name']}</td>
                        <td>{row['Department']}</td>
                        <td>{row['Satisfaction_Score']:.1f}</td>
                        <td>{row['Performance_Score']:.1f}</td>
                        <td style="color: red; font-weight: bold;">{row['Retention_Risk_Index']:.2f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
                <br>
                <p><strong>Recommended Actions:</strong></p>
                <ul>
                    <li>Schedule 1-on-1 conversations within 7 days</li>
                    <li>Review compensation competitiveness</li>
                    <li>Assess workload distribution</li>
                    <li>Monitor satisfaction scores for trends</li>
                </ul>
            </body>
        </html>
        """
        
        msg.set_content(html_content, subtype='html')
        
        # Send email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, password)
            smtp.send_message(msg)
        
        return True, f"Alert sent successfully to {receiver_email}"
    except Exception as e:
        return False, f"Failed to send alert: {str(e)}"

# Dashboard Layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Employee Analytics Dashboard", className="text-center text-primary mb-4"),
            html.P("Diagnostic Analytics for Workforce Management", className="text-center text-muted")
        ])
    ]),
    
    # Filters Section
    dbc.Row([
        dbc.Col([
            html.Label("Department:"),
            dcc.Dropdown(
                id='department-filter',
                options=[{'label': dept, 'value': dept} for dept in ['All'] + sorted(df['Department'].unique().tolist())],
                value='All',
                multi=False
            )
        ], width=3),
        
        dbc.Col([
            html.Label("Job Title:"),
            dcc.Dropdown(
                id='job-title-filter',
                options=[{'label': title, 'value': title} for title in ['All'] + sorted(df['Job_Title'].unique().tolist())],
                value='All',
                multi=False
            )
        ], width=3),
        
        dbc.Col([
            html.Label("Hire Year:"),
            dcc.Dropdown(
                id='hire-year-filter',
                options=[{'label': str(year), 'value': year} for year in ['All'] + sorted(df['Hire_Year'].unique().tolist())],
                value='All',
                multi=False
            )
        ], width=2),
        
        dbc.Col([
            html.Br(),
            dbc.Button("Reset Filters", id="reset-btn", color="secondary", className="mt-2")
        ], width=2),
        
        dbc.Col([
            html.Br(),
            dbc.Button("Send Alert", id="alert-btn", color="danger", className="mt-2")
        ], width=2)
    ], className="mb-4"),
    
    # Alert status
    dbc.Row([
        dbc.Col([
            html.Div(id='alert-status', className="text-center")
        ])
    ], className="mb-3"),
    
    # KPI Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Total Employees", className="card-title"),
                    html.H2(id='kpi-total-employees', className="text-primary")
                ])
            ])
        ], width=2),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Avg Productivity", className="card-title"),
                    html.H2(id='kpi-productivity', className="text-success")
                ])
            ])
        ], width=2),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Avg Satisfaction", className="card-title"),
                    html.H2(id='kpi-satisfaction', className="text-info")
                ])
            ])
        ], width=2),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Avg Age", className="card-title"),
                    html.H2(id='kpi-age', className="text-warning")
                ])
            ])
        ], width=2),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Avg Salary", className="card-title"),
                    html.H2(id='kpi-salary', className="text-danger")
                ])
            ])
        ], width=2),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Avg Tenure", className="card-title"),
                    html.H2(id='kpi-tenure', className="text-secondary")
                ])
            ])
        ], width=2)
    ], className="mb-4"),
    
    # Visualizations
    dbc.Row([
        dbc.Col([dcc.Graph(id='donut-chart')], width=4),
        dbc.Col([dcc.Graph(id='bar-chart')], width=4),
        dbc.Col([dcc.Graph(id='line-chart')], width=4)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([dcc.Graph(id='scatter-chart')], width=4),
        dbc.Col([dcc.Graph(id='box-chart')], width=4),
        dbc.Col([dcc.Graph(id='heatmap-chart')], width=4)
    ], className="mb-4"),
    
    # Chatbot Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Smart Chatbot - Ask Questions")),
                dbc.CardBody([
                    dbc.Textarea(
                        id='chatbot-input',
                        placeholder='Ask me anything... (e.g., "What is the average salary?", "Show performance by department", "Top 5 performers")',
                        style={'width': '100%', 'height': '80px'}
                    ),
                    html.Br(),
                    dbc.Button("Ask", id="chatbot-submit", color="primary", className="me-2"),
                    html.Hr(),
                    html.Div(id='chatbot-output', style={'white-space': 'pre-wrap', 'min-height': '150px', 'padding': '10px', 'background-color': '#f8f9fa', 'border-radius': '5px'})
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Auto-refresh interval
    dcc.Interval(id='interval-component', interval=60*1000, n_intervals=0)
    
], fluid=True)

# Callbacks
@app.callback(
    [Output('kpi-total-employees', 'children'),
     Output('kpi-productivity', 'children'),
     Output('kpi-satisfaction', 'children'),
     Output('kpi-age', 'children'),
     Output('kpi-salary', 'children'),
     Output('kpi-tenure', 'children'),
     Output('donut-chart', 'figure'),
     Output('bar-chart', 'figure'),
     Output('line-chart', 'figure'),
     Output('scatter-chart', 'figure'),
     Output('box-chart', 'figure'),
     Output('heatmap-chart', 'figure')],
    [Input('department-filter', 'value'),
     Input('job-title-filter', 'value'),
     Input('hire-year-filter', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_dashboard(dept, job, year, n):
    """Update all dashboard components based on filters"""
    filtered_df = df.copy()
    
    # Apply filters
    if dept != 'All':
        filtered_df = filtered_df[filtered_df['Department'] == dept]
    if job != 'All':
        filtered_df = filtered_df[filtered_df['Job_Title'] == job]
    if year != 'All':
        filtered_df = filtered_df[filtered_df['Hire_Year'] == year]
    
    # Calculate KPIs
    total_employees = len(filtered_df)
    avg_productivity = f"{filtered_df['Productivity'].mean():.1f}" if 'Productivity' in filtered_df.columns else "N/A"
    avg_satisfaction = f"{filtered_df['Satisfaction_Score'].mean():.1f}" if 'Satisfaction_Score' in filtered_df.columns else "N/A"
    avg_age = f"{filtered_df['Age'].mean():.0f}" if 'Age' in filtered_df.columns else "N/A"
    avg_salary = f"₹{filtered_df['Annual_Salary'].mean()/100000:.1f}L" if 'Annual_Salary' in filtered_df.columns else "N/A"
    avg_tenure = f"{filtered_df['Years_At_Company'].mean():.1f} yrs" if 'Years_At_Company' in filtered_df.columns else "N/A"
    
    # Donut Chart - Remote Work by Department
    remote_counts = filtered_df.groupby('Remote_Work_Frequency').size().reset_index(name='count')
    donut_fig = px.pie(remote_counts, values='count', names='Remote_Work_Frequency', 
                       title='Remote Work Distribution', hole=0.4)
    
    # Bar Chart - Projects by Job Title
    projects_by_job = filtered_df.groupby('Job_Title')['Projects_Handled'].mean().reset_index()
    bar_fig = px.bar(projects_by_job, x='Projects_Handled', y='Job_Title', 
                     title='Avg Projects by Job Title', orientation='h')
    
    # Line Chart - Performance over Tenure
    perf_by_tenure = filtered_df.groupby('Years_At_Company')['Performance_Score'].mean().reset_index()
    line_fig = px.line(perf_by_tenure, x='Years_At_Company', y='Performance_Score',
                       title='Performance vs Tenure', markers=True)
    
    # Scatter Plot - Age vs Performance
    scatter_fig = px.scatter(filtered_df, x='Age', y='Performance_Score', 
                            color='Department', title='Age vs Performance',
                            hover_data=['Name', 'Job_Title'])
    
    # Box Plot - Satisfaction by Department
    box_fig = px.box(filtered_df, x='Department', y='Satisfaction_Score',
                     title='Satisfaction Distribution by Department')
    
    # Heatmap - Performance vs Satisfaction
    heatmap_data = filtered_df.pivot_table(
        values='Employee_ID', 
        index=pd.cut(filtered_df['Performance_Score'], bins=5),
        columns=pd.cut(filtered_df['Satisfaction_Score'], bins=5),
        aggfunc='count',
        fill_value=0
    )
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=[str(col) for col in heatmap_data.columns],
        y=[str(idx) for idx in heatmap_data.index],
        colorscale='Viridis'
    ))
    heatmap_fig.update_layout(title='Performance vs Satisfaction Heatmap',
                             xaxis_title='Satisfaction Score Range',
                             yaxis_title='Performance Score Range')
    
    return (total_employees, avg_productivity, avg_satisfaction, avg_age, 
            avg_salary, avg_tenure, donut_fig, bar_fig, line_fig, 
            scatter_fig, box_fig, heatmap_fig)

@app.callback(
    Output('chatbot-output', 'children'),
    [Input('chatbot-submit', 'n_clicks')],
    [State('chatbot-input', 'value'),
     State('department-filter', 'value'),
     State('job-title-filter', 'value'),
     State('hire-year-filter', 'value')]
)
def handle_chatbot(n_clicks, query, dept, job, year):
    """Handle chatbot queries"""
    if not n_clicks or not query:
        return "Ask me a question about the employee data..."
    
    # Apply same filters as dashboard
    filtered_df = df.copy()
    if dept != 'All':
        filtered_df = filtered_df[filtered_df['Department'] == dept]
    if job != 'All':
        filtered_df = filtered_df[filtered_df['Job_Title'] == job]
    if year != 'All':
        filtered_df = filtered_df[filtered_df['Hire_Year'] == year]
    
    response = process_chatbot_query(query, filtered_df)
    return response

@app.callback(
    Output('alert-status', 'children'),
    [Input('alert-btn', 'n_clicks')]
)
def send_alert(n_clicks):
    """Send email alert for high-risk employees"""
    if not n_clicks:
        return ""
    
    high_risk = df[df['Retention_Risk_Index'] > 1.5]
    
    if len(high_risk) == 0:
        return dbc.Alert("No high-risk employees found.", color="success")
    
    success, message = send_email_alert(high_risk)
    
    if success:
        return dbc.Alert(f"✓ {message}", color="success")
    else:
        return dbc.Alert(f"✗ {message}", color="warning")

@app.callback(
    [Output('department-filter', 'value'),
     Output('job-title-filter', 'value'),
     Output('hire-year-filter', 'value')],
    [Input('reset-btn', 'n_clicks')]
)
def reset_filters(n_clicks):
    """Reset all filters to default"""
    if n_clicks:
        return 'All', 'All', 'All'
    return 'All', 'All', 'All'

if __name__ == '__main__':
    # FIXED: Changed from app.run_server to app.run to fix deprecation warning
    app.run(debug=True, port=9116)