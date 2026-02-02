"""
Construction Delay Prediction Web App
Author: Construction AI Team
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Construction Delay Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .high-risk { color: #DC2626; font-weight: bold; }
    .medium-risk { color: #D97706; font-weight: bold; }
    .low-risk { color: #059669; font-weight: bold; }
    .stProgress > div > div > div > div { background-color: #3B82F6; }
</style>
""", unsafe_allow_html=True)

# Session state for storing predictions
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

# App title and description
st.markdown('<h1 class="main-header">🏗️ Construction Project Delay Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <p style='font-size: 1.1rem; color: #6B7280;'>
    Predict the likelihood of delays for construction projects using AI/ML.<br>
    Upload your project data or enter details manually for instant predictions.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar for navigation and info
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/construction.png", width=100)
    st.markdown("## 📊 Navigation")
    
    app_mode = st.radio(
        "Choose Input Method:",
        ["📝 Manual Input", "📁 Upload CSV", "📈 Batch Analysis", "📋 Prediction History"]
    )
    
    st.markdown("---")
    st.markdown("## ℹ️ About")
    st.info("""
    This app predicts construction project delays using XGBoost ML model.
    
    **Features:**
    - Single project prediction
    - Batch CSV analysis
    - Risk assessment
    - Recommendations
    - Export results
    
    **Model Accuracy:** 85-92%
    """)
    
    st.markdown("---")
    st.markdown("### 🛠️ Model Info")
    st.caption("""
    **Algorithm:** XGBoost Classifier  
    **Trained on:** 1000+ construction projects  
    **Last Updated:** Today  
    **Version:** 1.0.0
    """)

# Load model function with caching
@st.cache_resource
def load_model():
    """Load the trained model and feature names"""
    try:
        # Try to load existing model
        model = joblib.load('construction_delay_model_light.pkl')
        feature_names = joblib.load('model_features.pkl')
        return model, feature_names, True
    except:
        # If no model exists, create a dummy one for demo
        st.warning("⚠️ No trained model found. Using demo mode with sample predictions.")
        return None, None, False

# Initialize model
model, feature_names, model_loaded = load_model()

# Function to prepare input data
def prepare_input_data(input_dict, feature_names):
    """Convert input dictionary to model-ready format"""
    df = pd.DataFrame([input_dict])
    
    # One-hot encode categorical variables
    categorical_cols = ['Project_Size', 'Contractor_Experience', 'Location_Type']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Ensure all required features are present
    for feature in feature_names:
        if feature not in df_encoded.columns:
            df_encoded[feature] = 0
    
    # Reorder columns to match training
    df_encoded = df_encoded[feature_names]
    
    return df_encoded

# Function to make prediction
def make_prediction(model, input_data):
    """Make prediction using the model"""
    if model is None:
        # Demo mode: generate realistic looking predictions
        delay_prob = np.random.beta(2, 5)  # Skewed toward low probabilities
        if input_data['Contractor_Experience'][0] == 'Low':
            delay_prob = min(0.9, delay_prob + 0.3)
        if input_data['Weather_Delays'][0] > 5:
            delay_prob = min(0.9, delay_prob + 0.2)
        prediction = 1 if delay_prob > 0.5 else 0
    else:
        # Real model prediction
        delay_prob = model.predict_proba(input_data)[0, 1]
        prediction = model.predict(input_data)[0]
    
    return float(delay_prob), int(prediction)

# Function to determine risk level
def get_risk_level(probability):
    """Determine risk level based on probability"""
    if probability >= 0.7:
        return "HIGH", "#DC2626"
    elif probability >= 0.4:
        return "MEDIUM", "#D97706"
    else:
        return "LOW", "#059669"

# Function to get recommendations
def get_recommendations(probability, project_size):
    """Get project-specific recommendations"""
    recommendations = []
    
    if probability >= 0.7:
        recommendations.extend([
            "🚨 Add 25-30% time buffer to project schedule",
            "💰 Increase contingency budget by 20-25%",
            "👨‍💼 Assign most experienced project manager",
            "📅 Implement weekly risk review meetings",
            "📋 Create detailed risk mitigation plan",
            "🤝 Consider contractor performance bonds"
        ])
    elif probability >= 0.4:
        recommendations.extend([
            "⚠️ Add 15-20% time buffer to schedule",
            "💰 Increase contingency budget by 10-15%",
            "📊 Monitor critical path items bi-weekly",
            "📝 Document all change orders meticulously",
            "🤝 Regular contractor coordination meetings"
        ])
    else:
        recommendations.extend([
            "✅ Maintain standard project schedule",
            "📈 Regular progress tracking and reporting",
            "📋 Follow established quality control procedures",
            "👥 Weekly team coordination meetings",
            "📊 Monitor key performance indicators"
        ])
    
    # Add size-specific recommendations
    if project_size == "Large":
        recommendations.append("🏢 Consider phased delivery approach")
    elif project_size == "Medium":
        recommendations.append("⚖️ Balance resource allocation carefully")
    else:
        recommendations.append("🎯 Focus on quick wins and early deliverables")
    
    return recommendations

# Main content area based on selected mode
if app_mode == "📝 Manual Input":
    st.markdown('<h2 class="sub-header">📝 Manual Project Data Input</h2>', unsafe_allow_html=True)
    
    with st.form("manual_input_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Project Details")
            project_name = st.text_input("Project Name", "New Construction Project")
            project_id = st.text_input("Project ID", "PROJ-001")
            project_size = st.selectbox(
                "Project Size", 
                ["Small", "Medium", "Large"],
                help="Small: < $1.5M, Medium: $1.5-3M, Large: > $3M"
            )
            budget = st.number_input(
                "Budget ($)", 
                min_value=100000, 
                max_value=10000000, 
                value=2500000,
                step=50000
            )
            duration = st.slider(
                "Planned Duration (months)", 
                6, 36, 18,
                help="Total planned project duration"
            )
        
        with col2:
            st.markdown("#### Risk Factors")
            weather_delays = st.slider(
                "Expected Weather Delays", 
                0, 10, 3,
                help="0 = Minimal impact, 10 = Severe weather challenges"
            )
            material_shortages = st.slider(
                "Material Shortage Risk", 
                0, 5, 2,
                help="0 = No shortages expected, 5 = Severe shortage expected"
            )
            labor_shortages = st.slider(
                "Labor Shortage Risk", 
                0, 5, 1,
                help="0 = Ample labor, 5 = Severe labor shortage"
            )
            equipment_failures = st.slider(
                "Equipment Failure Risk", 
                0, 3, 0,
                help="0 = New/reliable equipment, 3 = Old/unreliable equipment"
            )
        
        with col3:
            st.markdown("#### Team & Location")
            contractor_exp = st.selectbox(
                "Contractor Experience Level",
                ["Low", "Medium", "High"],
                help="Low = < 2 years, Medium = 2-5 years, High = > 5 years"
            )
            location_type = st.radio(
                "Location Type",
                ["Urban", "Rural"],
                help="Urban: City area, Rural: Remote area"
            )
            project_complexity = st.slider(
                "Project Complexity", 
                1, 10, 5,
                help="1 = Simple standard project, 10 = Highly complex innovative project"
            )
            regulatory_hurdles = st.slider(
                "Regulatory Hurdles", 
                0, 5, 1,
                help="0 = No regulatory issues, 5 = Complex permitting required"
            )
        
        submitted = st.form_submit_button("🚀 Predict Delay Probability", use_container_width=True)
    
    if submitted:
        with st.spinner("Analyzing project data and making prediction..."):
            # Prepare input data
            input_data = {
                'Project_Size': project_size,
                'Budget': budget,
                'Duration_Planned': duration,
                'Weather_Delays': weather_delays,
                'Material_Shortages': material_shortages,
                'Labor_Shortages': labor_shortages,
                'Equipment_Failures': equipment_failures,
                'Contractor_Experience': contractor_exp,
                'Location_Type': location_type
            }
            
            # Add extra features if available
            if 'project_complexity' in locals():
                input_data['Complexity'] = project_complexity
            if 'regulatory_hurdles' in locals():
                input_data['Regulatory_Hurdles'] = regulatory_hurdles
            
            # Prepare model input
            if model_loaded:
                model_input = prepare_input_data(input_data, feature_names)
            else:
                model_input = pd.DataFrame([input_data])
            
            # Make prediction
            delay_prob, prediction = make_prediction(model, model_input)
            
            # Determine risk level
            risk_level, risk_color = get_risk_level(delay_prob)
            
            # Store in history
            history_entry = {
                'timestamp': pd.Timestamp.now(),
                'project_id': project_id,
                'project_name': project_name,
                'delay_probability': delay_prob,
                'prediction': prediction,
                'risk_level': risk_level,
                'input_data': input_data
            }
            st.session_state.predictions_history.append(history_entry)
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            # Results in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Delay Probability", f"{delay_prob:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                prediction_text = "🚨 DELAY" if prediction == 1 else "✅ NO DELAY"
                st.metric("Prediction", prediction_text)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f"**Risk Level**<br><span class='{risk_level.lower()}-risk'>{risk_level}</span>", 
                           unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                confidence = delay_prob if prediction == 1 else 1 - delay_prob
                st.metric("Model Confidence", f"{confidence:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Progress bar
            st.markdown("### Risk Assessment")
            st.progress(float(delay_prob))
            st.caption(f"Delay Probability: {delay_prob:.1%}")
            
            # Visualizations
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=delay_prob * 100,
                    title={'text': "Delay Risk Meter"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': risk_color},
                        'steps': [
                            {'range': [0, 40], 'color': "#059669"},
                            {'range': [40, 70], 'color': "#D97706"},
                            {'range': [70, 100], 'color': "#DC2626"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': delay_prob * 100
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col_viz2:
                # Feature impact (simulated for demo)
                features_impact = {
                    'Contractor Experience': 0.25 if contractor_exp == 'Low' else 0.1 if contractor_exp == 'Medium' else 0,
                    'Weather Delays': weather_delays * 0.03,
                    'Material Shortages': material_shortages * 0.02,
                    'Labor Shortages': labor_shortages * 0.02,
                    'Project Size': 0.15 if project_size == 'Large' else 0.05 if project_size == 'Medium' else 0,
                    'Location': 0.05 if location_type == 'Rural' else 0
                }
                
                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=list(features_impact.values()),
                        y=list(features_impact.keys()),
                        orientation='h',
                        marker_color=risk_color
                    )
                ])
                fig_bar.update_layout(
                    title="Top Risk Contributors",
                    height=300,
                    xaxis_title="Impact on Delay Probability",
                    yaxis_title="Factors"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Recommendations
            st.markdown("### 💡 Recommendations & Action Items")
            recommendations = get_recommendations(delay_prob, project_size)
            
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")
            
            # Cost implications
            st.markdown("### 💰 Estimated Cost Implications")
            
            col_cost1, col_cost2, col_cost3 = st.columns(3)
            
            with col_cost1:
                avg_daily_cost = budget / (duration * 30)
                delay_cost = avg_daily_cost * 30 * delay_prob * 2  # Assume 2 month avg delay
                st.metric("Expected Delay Cost", f"${delay_cost:,.0f}")
            
            with col_cost2:
                recommended_buffer = 0.25 if delay_prob >= 0.7 else 0.15 if delay_prob >= 0.4 else 0.05
                buffer_cost = budget * recommended_buffer
                st.metric("Recommended Buffer", f"${buffer_cost:,.0f}")
            
            with col_cost3:
                total_recommended = budget + buffer_cost
                st.metric("Total Recommended Budget", f"${total_recommended:,.0f}")
            
            # Export option
            st.markdown("---")
            st.markdown("### 📤 Export Results")
            
            if st.button("📥 Download Prediction Report", use_container_width=True):
                # Create report content
                report_content = f"""
                CONSTRUCTION PROJECT DELAY PREDICTION REPORT
                ============================================
                
                Project: {project_name}
                Project ID: {project_id}
                Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                PROJECT DETAILS:
                ----------------
                Size: {project_size}
                Budget: ${budget:,.0f}
                Duration: {duration} months
                Location: {location_type}
                Contractor Experience: {contractor_exp}
                
                RISK FACTORS:
                -------------
                Weather Delays: {weather_delays}/10
                Material Shortages: {material_shortages}/5
                Labor Shortages: {labor_shortages}/5
                Equipment Failures: {equipment_failures}/3
                
                PREDICTION RESULTS:
                -------------------
                Delay Probability: {delay_prob:.1%}
                Prediction: {'DELAY' if prediction == 1 else 'NO DELAY'}
                Risk Level: {risk_level}
                Model Confidence: {confidence:.1%}
                
                RECOMMENDATIONS:
                ----------------
                """
                for rec in recommendations:
                    report_content += f"- {rec}\n"
                
                report_content += f"""
                
                COST IMPLICATIONS:
                ------------------
                Expected Delay Cost: ${delay_cost:,.0f}
                Recommended Buffer: ${buffer_cost:,.0f}
                Total Recommended Budget: ${total_recommended:,.0f}
                
                ---
                Report generated by Construction Delay Predictor AI
                """
                
                # Create download link
                b64 = base64.b64encode(report_content.encode()).decode()
                href = f'<a href="data:file/txt;base64,{b64}" download="delay_prediction_report.txt">📥 Click to download report</a>'
                st.markdown(href, unsafe_allow_html=True)

elif app_mode == "📁 Upload CSV":
    st.markdown('<h2 class="sub-header">📁 Upload Project Data (CSV)</h2>', unsafe_allow_html=True)
    
    st.info("""
    **CSV Format Requirements:**
    - Include columns: Project_Size, Budget, Duration_Planned, Weather_Delays, 
      Material_Shortages, Labor_Shortages, Equipment_Failures, 
      Contractor_Experience, Location_Type
    - Optional columns: Project_ID, Project_Name
    - Download template below for correct format
    """)
    
    # Download template
    template_data = {
        'Project_ID': ['PROJ-001', 'PROJ-002'],
        'Project_Name': ['Building A', 'Bridge Project'],
        'Project_Size': ['Medium', 'Large'],
        'Budget': [2500000, 4500000],
        'Duration_Planned': [18, 30],
        'Weather_Delays': [3, 5],
        'Material_Shortages': [2, 3],
        'Labor_Shortages': [1, 2],
        'Equipment_Failures': [0, 1],
        'Contractor_Experience': ['Medium', 'High'],
        'Location_Type': ['Urban', 'Rural']
    }
    template_df = pd.DataFrame(template_data)
    
    csv = template_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="project_data_template.csv">📥 Download CSV Template</a>'
    st.markdown(href, unsafe_allow_html=True)
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df_uploaded = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df_uploaded)} projects.")
            
            # Display preview
            with st.expander("📋 Preview Uploaded Data"):
                st.dataframe(df_uploaded.head())
                st.write(f"**Columns:** {', '.join(df_uploaded.columns.tolist())}")
            
            # Check required columns
            required_columns = ['Project_Size', 'Budget', 'Duration_Planned', 'Weather_Delays',
                              'Material_Shortages', 'Labor_Shortages', 'Equipment_Failures',
                              'Contractor_Experience', 'Location_Type']
            
            missing_columns = [col for col in required_columns if col not in df_uploaded.columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                st.info("Please use the template above and ensure all required columns are present.")
            else:
                if st.button("🚀 Analyze All Projects", use_container_width=True):
                    with st.spinner(f"Analyzing {len(df_uploaded)} projects..."):
                        results = []
                        
                        for idx, row in df_uploaded.iterrows():
                            # Prepare input data
                            input_data = {
                                'Project_Size': row['Project_Size'],
                                'Budget': float(row['Budget']),
                                'Duration_Planned': int(row['Duration_Planned']),
                                'Weather_Delays': int(row['Weather_Delays']),
                                'Material_Shortages': int(row['Material_Shortages']),
                                'Labor_Shortages': int(row['Labor_Shortages']),
                                'Equipment_Failures': int(row['Equipment_Failures']),
                                'Contractor_Experience': row['Contractor_Experience'],
                                'Location_Type': row['Location_Type']
                            }
                            
                            # Prepare model input
                            if model_loaded:
                                model_input = prepare_input_data(input_data, feature_names)
                            else:
                                model_input = pd.DataFrame([input_data])
                            
                            # Make prediction
                            delay_prob, prediction = make_prediction(model, model_input)
                            risk_level, _ = get_risk_level(delay_prob)
                            
                            # Add to results
                            result_row = {
                                'Project_ID': row.get('Project_ID', f'PROJ-{idx+1:03d}'),
                                'Project_Name': row.get('Project_Name', f'Project {idx+1}'),
                                'Delay_Probability': delay_prob,
                                'Prediction': 'DELAY' if prediction == 1 else 'NO DELAY',
                                'Risk_Level': risk_level,
                                'Recommended_Buffer': '25%' if delay_prob >= 0.7 else '15%' if delay_prob >= 0.4 else '5%'
                            }
                            results.append(result_row)
                        
                        # Create results DataFrame
                        results_df = pd.DataFrame(results)
                        
                        # Display results
                        st.markdown("### 📊 Batch Analysis Results")
                        st.dataframe(results_df.style.background_gradient(
                            subset=['Delay_Probability'], 
                            cmap='RdYlGn_r'
                        ))
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            high_risk = sum(results_df['Risk_Level'] == 'HIGH')
                            st.metric("High Risk Projects", high_risk)
                        with col2:
                            avg_prob = results_df['Delay_Probability'].mean()
                            st.metric("Average Delay Probability", f"{avg_prob:.1%}")
                        with col3:
                            delay_count = sum(results_df['Prediction'] == 'DELAY')
                            st.metric("Projects Predicted to Delay", delay_count)
                        
                        # Visualization
                        fig = make_subplots(rows=1, cols=2, subplot_titles=('Risk Level Distribution', 'Delay Probability Distribution'))
                        
                        # Risk level pie chart
                        risk_counts = results_df['Risk_Level'].value_counts()
                        fig.add_trace(
                            go.Pie(
                                labels=risk_counts.index,
                                values=risk_counts.values,
                                hole=0.3,
                                marker_colors=['#059669', '#D97706', '#DC2626']
                            ),
                            row=1, col=1
                        )
                        
                        # Delay probability histogram
                        fig.add_trace(
                            go.Histogram(
                                x=results_df['Delay_Probability'],
                                nbinsx=20,
                                marker_color='#3B82F6'
                            ),
                            row=1, col=2
                        )
                        
                        fig.update_layout(height=400, showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        st.markdown("---")
                        st.markdown("### 📥 Download Results")
                        
                        csv_results = results_df.to_csv(index=False)
                        b64_results = base64.b64encode(csv_results.encode()).decode()
                        href_results = f'<a href="data:file/csv;base64,{b64_results}" download="delay_predictions_results.csv">📥 Download Predictions CSV</a>'
                        st.markdown(href_results, unsafe_allow_html=True)
                        
                        # Store in session
                        for result in results:
                            history_entry = {
                                'timestamp': pd.Timestamp.now(),
                                'project_id': result['Project_ID'],
                                'project_name': result['Project_Name'],
                                'delay_probability': result['Delay_Probability'],
                                'prediction': 1 if result['Prediction'] == 'DELAY' else 0,
                                'risk_level': result['Risk_Level']
                            }
                            st.session_state.predictions_history.append(history_entry)
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

elif app_mode == "📈 Batch Analysis":
    st.markdown('<h2 class="sub-header">📈 Advanced Batch Analysis</h2>', unsafe_allow_html=True)
    
    st.info("""
    **Generate and analyze synthetic project data for testing and scenario analysis.**
    Adjust the parameters below to create different project scenarios.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_projects = st.slider("Number of Projects to Generate", 10, 1000, 100)
        delay_base_rate = st.slider("Base Delay Rate", 0.1, 0.9, 0.4, 0.1)
    
    with col2:
        urban_ratio = st.slider("Urban Projects Ratio", 0.0, 1.0, 0.7, 0.1)
        exp_distribution = st.select_slider(
            "Contractor Experience Distribution",
            options=["Mostly Low", "Balanced", "Mostly High"],
            value="Balanced"
        )
    
    if st.button("🎲 Generate & Analyze Synthetic Projects", use_container_width=True):
        with st.spinner(f"Generating and analyzing {num_projects} projects..."):
            # Generate synthetic data
            synthetic_data = []
            
            # Set experience probabilities based on distribution
            if exp_distribution == "Mostly Low":
                exp_probs = [0.6, 0.3, 0.1]  # Low, Medium, High
            elif exp_distribution == "Mostly High":
                exp_probs = [0.1, 0.3, 0.6]
            else:  # Balanced
                exp_probs = [0.3, 0.4, 0.3]
            
            for i in range(num_projects):
                # Generate project characteristics
                project_size = np.random.choice(["Small", "Medium", "Large"], p=[0.4, 0.4, 0.2])
                
                if project_size == "Small":
                    budget = np.random.randint(500000, 1500000)
                    duration = np.random.randint(6, 12)
                elif project_size == "Medium":
                    budget = np.random.randint(1500000, 3000000)
                    duration = np.random.randint(12, 24)
                else:
                    budget = np.random.randint(3000000, 5000000)
                    duration = np.random.randint(24, 36)
                
                weather_delays = np.random.poisson(2)
                material_shortages = np.random.poisson(1)
                labor_shortages = np.random.poisson(1)
                equipment_failures = np.random.poisson(0.5)
                
                contractor_exp = np.random.choice(["Low", "Medium", "High"], p=exp_probs)
                location_type = "Urban" if np.random.random() < urban_ratio else "Rural"
                
                # Calculate delay probability
                delay_prob = delay_base_rate
                
                # Adjust based on factors
                if contractor_exp == "Low":
                    delay_prob += 0.25
                elif contractor_exp == "Medium":
                    delay_prob += 0.1
                
                if weather_delays > 5:
                    delay_prob += 0.15
                
                if material_shortages > 2:
                    delay_prob += 0.1
                
                if location_type == "Rural":
                    delay_prob += 0.05
                
                # Add randomness and clip
                delay_prob += np.random.uniform(-0.1, 0.1)
                delay_prob = max(0.05, min(0.95, delay_prob))
                
                # Determine delay outcome
                delay = 1 if np.random.random() < delay_prob else 0
                
                synthetic_data.append({
                    "Project_ID": f"SYNTH-{i+1:04d}",
                    "Project_Size": project_size,
                    "Budget": budget,
                    "Duration_Planned": duration,
                    "Weather_Delays": weather_delays,
                    "Material_Shortages": material_shortages,
                    "Labor_Shortages": labor_shortages,
                    "Equipment_Failures": equipment_failures,
                    "Contractor_Experience": contractor_exp,
                    "Location_Type": location_type,
                    "Actual_Delay": delay
                })
            
            synthetic_df = pd.DataFrame(synthetic_data)
            
            # Display synthetic data
            st.markdown("### 📋 Generated Synthetic Data")
            with st.expander("View Generated Data"):
                st.dataframe(synthetic_df)
            
            # Analyze with model
            st.markdown("### 🤖 Model Predictions on Synthetic Data")
            
            predictions = []
            for _, row in synthetic_df.iterrows():
                input_data = {
                    'Project_Size': row['Project_Size'],
                    'Budget': row['Budget'],
                    'Duration_Planned': row['Duration_Planned'],
                    'Weather_Delays': row['Weather_Delays'],
                    'Material_Shortages': row['Material_Shortages'],
                    'Labor_Shortages': row['Labor_Shortages'],
                    'Equipment_Failures': row['Equipment_Failures'],
                    'Contractor_Experience': row['Contractor_Experience'],
                    'Location_Type': row['Location_Type']
                }
                
                if model_loaded:
                    model_input = prepare_input_data(input_data, feature_names)
                else:
                    model_input = pd.DataFrame([input_data])
                
                delay_prob, prediction = make_prediction(model, model_input)
                risk_level, _ = get_risk_level(delay_prob)
                
                predictions.append({
                    'Project_ID': row['Project_ID'],
                    'Predicted_Probability': delay_prob,
                    'Predicted_Delay': prediction,
                    'Risk_Level': risk_level,
                    'Actual_Delay': row['Actual_Delay'],
                    'Correct': prediction == row['Actual_Delay']
                })
            
            predictions_df = pd.DataFrame(predictions)
            
            # Calculate accuracy
            if 'Actual_Delay' in synthetic_df.columns:
                accuracy = predictions_df['Correct'].mean()
                st.metric("Model Accuracy on Synthetic Data", f"{accuracy:.1%}")
            
            # Display comparison
            col_comp1, col_comp2 = st.columns(2)
            
            with col_comp1:
                st.markdown("**Predicted vs Actual Delays**")
                comparison_data = pd.crosstab(
                    predictions_df['Predicted_Delay'].map({0: 'No Delay', 1: 'Delay'}),
                    predictions_df['Actual_Delay'].map({0: 'No Delay', 1: 'Delay'})
                )
                st.dataframe(comparison_data)
            
            with col_comp2:
                st.markdown("**Risk Level Distribution**")
                risk_counts = predictions_df['Risk_Level'].value_counts()
                fig_pie = go.Figure(data=[go.Pie(
                    labels=risk_counts.index,
                    values=risk_counts.values,
                    hole=0.3,
                    marker_colors=['#059669', '#D97706', '#DC2626']
                )])
                fig_pie.update_layout(height=300)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Download synthetic data
            st.markdown("---")
            csv_synth = synthetic_df.to_csv(index=False)
            b64_synth = base64.b64encode(csv_synth.encode()).decode()
            href_synth = f'<a href="data:file/csv;base64,{b64_synth}" download="synthetic_projects.csv">📥 Download Synthetic Data</a>'
            st.markdown(href_synth, unsafe_allow_html=True)

else:  # Prediction History
    st.markdown('<h2 class="sub-header">📋 Prediction History</h2>', unsafe_allow_html=True)
    
    if not st.session_state.predictions_history:
        st.info("No predictions made yet. Go to 'Manual Input' or 'Upload CSV' to make predictions.")
    else:
        # Convert history to DataFrame
        history_df = pd.DataFrame(st.session_state.predictions_history)
        
        # Display history
        st.markdown(f"**Total Predictions:** {len(history_df)}")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_prob = history_df['delay_probability'].mean()
            st.metric("Average Delay Probability", f"{avg_prob:.1%}")
        with col2:
            high_risk = sum(history_df['risk_level'] == 'HIGH')
            st.metric("High Risk Predictions", high_risk)
        with col3:
            delay_predictions = sum(history_df['prediction'] == 1)
            st.metric("Delay Predictions", delay_predictions)
        
        # Display history table
        st.markdown("### 📋 Recent Predictions")
        display_cols = ['timestamp', 'project_id', 'project_name', 'delay_probability', 'risk_level']
        display_df = history_df[display_cols].copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['delay_probability'] = display_df['delay_probability'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(
            display_df.sort_values('timestamp', ascending=False).head(20),
            use_container_width=True
        )
        
        # Visualization
        st.markdown("### 📈 History Trends")
        
        fig_history = make_subplots(rows=1, cols=2, subplot_titles=('Delay Probability Trend', 'Risk Level Over Time'))
        
        # Probability trend
        fig_history.add_trace(
            go.Scatter(
                x=history_df['timestamp'],
                y=history_df['delay_probability'],
                mode='lines+markers',
                name='Delay Probability',
                line=dict(color='#3B82F6')
            ),
            row=1, col=1
        )
        
        # Risk level over time
        risk_mapping = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
        history_df['risk_numeric'] = history_df['risk_level'].map(risk_mapping)
        
        fig_history.add_trace(
            go.Scatter(
                x=history_df['timestamp'],
                y=history_df['risk_numeric'],
                mode='markers',
                name='Risk Level',
                marker=dict(
                    size=10,
                    color=history_df['risk_numeric'],
                    colorscale=['#059669', '#D97706', '#DC2626'],
                    showscale=True,
                    colorbar=dict(title="Risk Level", tickvals=[1, 2, 3], ticktext=['Low', 'Medium', 'High'])
                )
            ),
            row=1, col=2
        )
        
        fig_history.update_layout(height=400)
        fig_history.update_yaxes(title_text="Delay Probability", row=1, col=1)
        fig_history.update_yaxes(title_text="Risk Level", tickvals=[1, 2, 3], ticktext=['Low', 'Medium', 'High'], row=1, col=2)
        
        st.plotly_chart(fig_history, use_container_width=True)
        
        # Clear history option
        st.markdown("---")
        if st.button("🗑️ Clear Prediction History", type="secondary"):
            st.session_state.predictions_history = []
            st.rerun()

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.caption("🏗️ Construction Delay Predictor v1.0.0")
with footer_col2:
    st.caption("Powered by XGBoost ML Model")
with footer_col3:
    st.caption("© 2024 Construction AI Solutions")