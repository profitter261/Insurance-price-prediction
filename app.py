import streamlit as st
import mysql.connector
import plotly.express as px
import pandas as pd

def get_connection():
    connection = mysql.connector.connect(
        host="127.0.0.1",
        port = "3306",
        password = "Arvind@2003",  # or your DB host (e.g., "127.0.0.1" or cloud endpoint)
        user="root",          # your MySQL username
        database="insurance"  # your database name
    )
    return connection

# Fetch data from MySQL
def load_data():
    conn = get_connection()
    query = "SELECT * FROM cleaned_dataset;"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def load_data1():
    conn = get_connection()
    query1 = "SELECT * FROM medical_insurance;"
    df1 = pd.read_sql(query1, conn)
    conn.close()
    return df1
    
data = load_data()
data1 = load_data1()
st.set_page_config(layout="wide")
st.title('Loan Prediction App')
pages = st.sidebar.selectbox('choose:', ['Home', 'Analysis', 'Prediction', 'Confidence Intervals'])

if pages == 'Home':
    with st.container():
        st.subheader("Project Introduction")
        st.write("This project focuses on predicting medical insurance charges using Machine Learning. Healthcare costs often vary significantly depending on factors such as a person’s age, gender, body mass index (BMI), smoking habits, number of dependents, and residential region. By leveraging data-driven approaches, this project aims to provide a reliable estimate of individual insurance costs, making the system more transparent and helping both users and insurers in decision-making.")
        st.write("This project demonstrates how machine learning can support real-world decision-making in healthcare and insurance domains, offering both analytical insights and practical applications through an easy-to-use interface.")
    with st.container():
        st.subheader('The workflow includes:')
        st.markdown(
         """
        - Data Preprocessing: Cleaning and preparing the insurance dataset
        - Feature Engineering: Categorizing variables like age and BMI into meaningful groups for better interpretability
        - Model Training: Applying regression-based machine learning algorithms (e.g., Linear Regression, XGBoost, etc.) to predict charges.
        - Evaluation: Measuring model performance with accuracy metrics and confidence intervals.
        - Deployment: Building an interactive Streamlit web application where users can input their personal details (age range, BMI category, smoking status, number of children, etc.) and get an instant prediction of insurance charges along with visual insights.
        """
        )
    with st.expander("View Dataset:", expanded=False):
        st.dataframe(data1)
           
if pages == 'Analysis':
    st.set_page_config(layout="wide")
    choices = st.sidebar.selectbox('which one ?', ['Univariate', 'Bivariate', 'multivariate', 'outlier analysis'])


    if choices == 'Univariate':
        st.markdown("## 📊 Univariate Analysis")
        st.markdown("Explore how individual variables are distributed across policyholders.")

    # --- Row 1: Smokers vs Non-Smokers + Region Distribution ---
        col1, col2 = st.columns(2)

        with col1:
            non_smoker_count = data[data['smoker_yes'] == 0]['smoker_yes'].count()
            smoker_count = data[data['smoker_yes'] == 1]['smoker_yes'].count()

            fig1 = px.bar(
            x=['Non-Smokers', 'Smokers'],
            y=[non_smoker_count, smoker_count],
            color=['Non-Smokers', 'Smokers'],
            text=[non_smoker_count, smoker_count],
            color_discrete_sequence=["#00C9A7", "#FF6B6B"]
            )
            fig1.update_layout(
            xaxis_title="Smoker Status",
            yaxis_title="Number of Policyholders",
            template="plotly_dark",
            showlegend=False
            )
            with st.container(border=True):
                st.subheader("🚬 Smokers vs Non-Smokers")
                st.plotly_chart(fig1, use_container_width=True)

        with col2:
            region_counts = {
            'Northwest': data['region_northwest'].sum(),
            'Southeast': data['region_southeast'].sum(),
            'Southwest': data['region_southwest'].sum()
            }
            max_region = max(region_counts, key=region_counts.get)

            fig2 = px.pie(
            values=region_counts.values(),
            names=region_counts.keys(),
            color_discrete_sequence=px.colors.sequential.Tealgrn
            )
            fig2.update_traces(textinfo="percent+label", pull=[0.05, 0.05, 0.05])
            fig2.update_layout(template="plotly_dark")
            with st.container(border=True):
                st.subheader("🌍 Region-Wise Distribution")
                st.plotly_chart(fig2, use_container_width=True)
                st.info(f"✅ Majority of policyholders are from **{max_region}** region.")

    # --- Row 2: Charges Distribution + Age Distribution ---
        col3, col5 = st.columns(2)

        with col3:
            fig3 = px.histogram(
            data,
            x='charges_log',
            nbins=40,
            color_discrete_sequence=["#36CFC9"]
            )
            fig3.update_layout(
            xaxis_title="Log of Charges",
            yaxis_title="Number of Policyholders",
            template="plotly_dark"
            )
            with st.container(border=True):
                st.subheader("💰 Distribution of Medical Insurance Charges")
                st.plotly_chart(fig3, use_container_width=True)

        with col5:
            dic_01 = {0: '0-18', 1: '18-26', 2: '26-45', 3: '45-65', 4: '65-100'}
            fig5 = px.histogram(
            data,
            x=data['age'].map(dic_01),
            color_discrete_sequence=["#A066FF"]
            )
            fig5.update_layout(
            xaxis_title="Age Group",
            yaxis_title="Number of Policyholders",
            template="plotly_dark"
            )
            with st.container(border=True):
                st.subheader("🎂 Age Distribution of Individuals")

            # KPI Cards
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("📏 Avg. BMI", round(data['bmi'].mean(), 2))
                kpi2.metric("💵 Avg. Charges", round(data['charges_log'].mean(), 2))
                kpi3.metric("👶 Avg. Children", round(data['children'].mean(), 2))

                st.plotly_chart(fig5, use_container_width=True)

        
    elif choices == 'Bivariate':
        st.markdown("## 🔗 Bivariate Analysis")
        st.markdown("Understand how two variables together influence insurance charges.")

    # --- Row 1: Age vs Charges | Smokers vs Non-Smokers ---
        col1, col2 = st.columns(2)

        with col1:
            dic_01 = {0: '0-18', 1: '18-26', 2: '26-45', 3: '45-65', 4: '65-100'}
            age_groups = data['age'].map(dic_01)
            fig6 = px.box(
            data,
            x=age_groups,
            y="charges_log",
            color=age_groups,
            color_discrete_sequence=px.colors.sequential.Viridis
            )
            fig6.update_layout(
            xaxis_title="Age Group",
            yaxis_title="Insurance Charges (log scale)",
            template="plotly_dark",
            showlegend=False
            )
            with st.container(border=True):
                st.subheader('🎂 Age vs Insurance Charges')
                st.plotly_chart(fig6, use_container_width=True)
                st.info(f"💡 Highest charges are concentrated in **{dic_01[max(data['age'])]}** group")

        with col2:
            smoker_charges = data1.groupby('smoker')['charges'].mean().reset_index()
            fig7 = px.bar(
            smoker_charges,
            x="smoker",
            y="charges",
            text="charges",
            color="smoker",
            color_discrete_sequence=["#00C9A7", "#FF6B6B"]
            )
            fig7.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig7.update_layout(
            xaxis_title="Smoker Status",
            yaxis_title="Average Charges",
            template="plotly_dark",
            showlegend=False
            )
            with st.container(border=True):
                st.subheader('🚬 Average Charges: Smokers vs Non-Smokers')
                st.plotly_chart(fig7, use_container_width=True)

    # --- Row 2: BMI vs Charges | Gender vs Charges ---
        col3, col5 = st.columns(2)

        with col3:
            fig8 = px.scatter(
            data,
            x="bmi",
            y="charges_log",
            color="smoker_yes",
            size="age",
            opacity=0.7,
            color_discrete_sequence=px.colors.qualitative.Set1,
            )
            fig8.update_layout(
            xaxis_title="BMI",
            yaxis_title="Insurance Charges (log scale)",
            template="plotly_dark"
            )
            with st.container(border=True):
                st.subheader("⚖️ BMI vs Insurance Charges")
                st.plotly_chart(fig8, use_container_width=True)

        with col5:
            gender_charges = data1.groupby('sex')['charges'].mean().reset_index()
            fig9 = px.bar(
            gender_charges,
            x="sex",
            y="charges",
            text="charges",
            color="sex",
            color_discrete_sequence=["#36CFC9", "#A066FF"]
            )
            fig9.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig9.update_layout(
            xaxis_title="Gender",
            yaxis_title="Average Charges",
            template="plotly_dark",
            showlegend=False
            )

        # BMI categorization
            def bmi_category(bmi):
                if bmi < 18.5:
                    return 'Low BMI'
                elif 18.5 <= bmi < 25:
                    return 'Healthy'
                elif 25 <= bmi < 30:
                    return 'Moderate'
                else:
                    return 'High BMI'

            with st.container(border=True):
                st.subheader("🧑‍🤝‍🧑 Gender vs Insurance Charges")
                st.plotly_chart(fig9, use_container_width=True)
                st.info(f"💡 Maximum charges are taken by policyholders with **{bmi_category(max(data['bmi']))}** BMI group.")

                  
    elif choices == 'multivariate':
        col1, col2 = st.columns(2)

    # Scatter: Smoking status + Age vs Charges
        with col1:
            fig9 = px.scatter(
            data,
            x="age",
            y="charges_log",
            color="smoker_yes",
            opacity=0.7,
            title="Impact of Smoking Status and Age on Medical Charges"
            )
            fig9.update_layout(
            xaxis_title="Age",
            yaxis_title="Log Charges",
            template="plotly_dark"
            )
            with st.container(border=True):
                st.plotly_chart(fig9, use_container_width=True)

    # Bar: Gender & Region for Smokers
        with col2:
            agg_data = (
            data1[data1['smoker'] == 'yes']
            .groupby(['region', 'sex'], as_index=False)['charges']
            .mean()
            )
            fig8 = px.bar(
            agg_data,
            x='region',
            y='charges',
            color='sex',
            barmode='group',
            title="Impact of Gender and Region on Charges for Smokers"
            )
            fig8.update_layout(
            xaxis_title="Region",
            yaxis_title="Average Charges",
            template="plotly_dark"
            )
            with st.container(border=True):
                st.plotly_chart(fig8, use_container_width=True)

        col3, col5 = st.columns(2)

    # Scatter: Age vs Charges with BMI as size
        with col3:
            fig10 = px.scatter(
            data,
            x="age",
            y="charges_log",
            size="bmi",
            color="smoker_yes",
            opacity=0.6,
            size_max=60,
            title="Distribution of Medical Insurance Charges"
            )
            fig10.update_layout(
            xaxis_title="Age",
            yaxis_title="Log Charges",
            template="plotly_dark"
            )
            with st.container(border=True):
                st.plotly_chart(fig10, use_container_width=True)

    # Heatmap: Correlation Matrix
        with col5:
            corr = data.corr(numeric_only=True)

            fig_corr = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            origin="lower",
            aspect="auto",
            title="Correlation Heatmap"
            )
            fig_corr.update_layout(
            xaxis_title="Features",
            yaxis_title="Features",
            template="plotly_dark"
            )
            with st.container(border=True):
                st.plotly_chart(fig_corr, use_container_width=True)

            
    elif choices == 'outlier analysis':
        col1, col2 = st.columns(2)

    # Boxplot: Insurance charges
        with col1:
            with st.container(border=True):
                fig = px.box(
                data1,
                y="charges",
                title="Outliers in Insurance Charges"
                )
                fig.update_layout(
                yaxis_title="Charges",
                template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)

    # Boxplot: BMI
        with col2:
            with st.container(border=True):
                fig = px.box(
                data1,
                y="bmi",
                title="Outliers in BMI Indices"
                )   
                fig.update_layout(
                yaxis_title="BMI",
                template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)

    # Expanders for detailed tables
        col3, col4, col5 = st.columns(3)

        with col3:
            highest_payers = data1.sort_values(by='charges', ascending=False).head(5)
            with st.expander('💰 Top 5 Policyholders with Highest Charges', expanded=True):
                st.dataframe(highest_payers)

        with col4:
            corr_matrix = data.corr(numeric_only=True)
            correlation_with_charges = (
            corr_matrix['charges_log']
            .drop('charges_log')
            .sort_values(ascending=False)
            )
            with st.expander('📊 Correlation of Insurance Charges with Features', expanded=True):
                st.dataframe(correlation_with_charges)

        with col5:
            Q1 = data1['bmi'].quantile(0.25)
            Q3 = data1['bmi'].quantile(0.75)
            IQR = Q3 - Q1
            bmi_outliers = data1[data1['bmi'] > (Q3 + 1.5 * IQR)]
            with st.expander('⚠️ Extreme BMI Values', expanded=True):
                st.dataframe(bmi_outliers)

    # Obese Smokers vs Non-Obese Non-Smokers comparison
        col7 = st.columns(1)
        with col7[0]:
            with st.container(border=True):
                st.subheader('Charges on Obese Smokers vs Non-Obese Non-Smokers')

                data1['obese'] = data1['bmi'] > 30
                comparison_df = data1[
                ((data1['obese']) & (data1['smoker'] == 'yes')) |
                ((~data1['obese']) & (data1['smoker'] == 'no'))
                ]

                comparison_df['Group'] = comparison_df.apply(
                lambda row: 'Obese Smokers' if row['obese'] else 'Non-Obese Non-Smokers',
                axis=1
                )

                fig = px.box(
                comparison_df,
                x="Group",
                y="charges",
                color="smoker",
                title="Charges: Obese Smokers vs Non-Obese Non-Smokers"
                )
                fig.update_layout(
                xaxis_title="Group",
                yaxis_title="Charges",
                template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)

            
    
if pages == 'Prediction':
    st.write("let's discuss about the Prediction")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            age = st.selectbox("Age", ['0-18', '19-35', '36-50', '51-65', '66-100'])
    
    with col2:
        with st.container(border=True):
            bmi = st.selectbox("BMI", ['less than 18.5', '18.5 to 24', '25 to 29', 'greater or equal to 30'])

    col3, col4 = st.columns(2)
    with col3:
        with st.container(border=True):
            gender = st.radio("Gender", ["male", "female"])

    with col4:
        with st.container(border=True):
            smoking_status = st.radio("Do you smoke?", ["yes", "no"])
        
    col5, col6 = st.columns(2)
    with col5:
        with st.container(border=True):
            children = st.select_slider("Number of Children", [0, 1, 2, 3, 4, 5])
        
    with col6:
        with st.container(border=True):
            # Initialize one-hot region values
            region_northwest = region_southeast = region_southwest = 0
            region = st.selectbox("Region", ["southeast", "southwest", "northwest"])

            if region == "northwest":
                region_northwest = 1
            elif region == "southeast":
                region_southeast = 1
            elif region == "southwest":
                region_southwest = 1

    clicked = st.button('Predict')
    
    if clicked:
        import numpy as np
        import joblib

        # Load trained model
        model = joblib.load("xgboost_model (1).pkl")

        # --- Input Transformation Functions ---
        def transform_age(age_range):
            mapping = {
                '0-18': 10,
                '19-35': 27,
                '36-50': 43,
                '51-65': 58,
                '66-100': 80
            }
            return mapping[age_range]

        def transform_bmi(bmi_range):
            mapping = {
                'less than 18.5': 17,
                '18.5 to 24': 21,
                '25 to 29': 27,
                'greater or equal to 30': 32
            }
            return mapping[bmi_range]

        def transform_gender(gender):
            return 1 if gender == "male" else 0

        def transform_smoking(smoke):
            return 1 if smoke == "yes" else 0

        # Transform inputs
        age_val = transform_age(age)
        bmi_val = transform_bmi(bmi)
        gender_val = transform_gender(gender)
        smoke_val = transform_smoking(smoking_status)
        children_val = children

        # Arrange features in same order as training
        features = np.array([[age_val, bmi_val, gender_val, smoke_val, children_val,
                              region_northwest, region_southeast, region_southwest]])

        # Make prediction
        prediction = model.predict(features)
        st.success(f"Predicted Insurance Cost: ${np.exp(prediction[0]):,.2f}")
        
if pages == 'Confidence Intervals':
    from sklearn.model_selection import train_test_split
    import numpy as np
    import xgboost as xgb
    import streamlit as st
    import plotly.graph_objs as go

# Get the list of features to plot against (excluding the constant)
    df = data
    X = df.drop(['charges_log', 'age_bmi_interaction', 'smoker_bmi_interaction'], axis=1)
    y = df['charges_log']
    preds = []
    xgbr = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, enable_categorical=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    xgbr.fit(X_train, y_train)
    for i in range(100):  # bootstrap 100 times
        sample = X.sample(frac=1, replace=True)
        sample_y = y.loc[sample.index] # Get corresponding target values
        xgbr.fit(sample, sample_y)
        preds.append(xgbr.predict(X_test))

    preds = np.array(preds)
    mean_preds = preds.mean(axis=0)
    ci_low = np.percentile(preds, 2.5, axis=0)
    ci_high = np.percentile(preds, 97.5, axis=0)
    
    # --- Streamlit UI ---
    st.subheader("Confidence Interval Visualization")

# Dropdown to select feature
    feature = st.selectbox("Select feature to plot:", X_test.columns)
    with st.container():
        st.subheader("What is a Confidence Interval (CI)?")
        st.write("A confidence interval is a range of values that is likely to contain the true value of something we are trying to estimate (in our case, the 'charges_log'). Instead of just giving a single predicted value, a CI provides a range, giving us a sense of the precision of the estimate.")
# Sort values for smooth line plots
    X_test_sorted = X_test.sort_values(by=feature)
    sorted_indices = X_test_sorted.index
    mean_preds_sorted = mean_preds[X_test.index.get_indexer(sorted_indices)]
    ci_low_sorted = ci_low[X_test.index.get_indexer(sorted_indices)]
    ci_high_sorted = ci_high[X_test.index.get_indexer(sorted_indices)]

# Plotly figure
    fig = go.Figure()

# Mean predictions
    fig.add_trace(go.Scatter(
    x=X_test_sorted[feature], y=mean_preds_sorted,
    mode='lines', name='Prediction'
    ))

# Confidence Interval
    fig.add_trace(go.Scatter(
    x=X_test_sorted[feature], y=ci_high_sorted,
    mode='lines', line=dict(width=0),
    showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
    x=X_test_sorted[feature], y=ci_low_sorted,
    mode='lines', line=dict(width=0),
    fill='tonexty', fillcolor='rgba(0,100,80,0.2)',
    name='95% CI'
    ))

    fig.update_layout(
    title=f'Predictions with 95% Confidence Interval vs. {feature.capitalize()}',
    xaxis_title=feature.capitalize(),
    yaxis_title='charges_log'
    )

# Show in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    
    