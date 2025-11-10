import streamlit as st
from streamlit_option_menu import option_menu
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import joblib
import numpy as np

# ========== PAGE CONFIG ==========
st.set_page_config(layout="wide", page_title="🏥 Insurance Prediction App")

# ========== LOAD DATA ==========
@st.cache_data
def load_data():
    return pd.read_csv("Dataset/cleaned_medical_insurance_data.csv")

@st.cache_data
def load_data1():
    return pd.read_csv("Dataset/medical_insurance.csv")

data = load_data()
data1 = load_data1()

# ========== MAIN LAYOUT ==========
st.title("🏥 Insurance Prediction App")

left_col, right_col = st.columns([1, 4], gap="large")

# -------------------- LEFT MENU --------------------
with left_col:
    selected = option_menu(
        menu_title="Main Menu",
        options=['Home', 'Analysis', 'Prediction', 'Model Performances'],
        icons=['house', 'bar-chart', 'cpu', 'activity'],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "0!important", "background-color": "#0E1117"},
            "icon": {"color": "#00B4D8", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px 0px",
                "--hover-color": "#262730",
            },
            "nav-link-selected": {"background-color": "#00B4D8"},
        },
    )

# -------------------- RIGHT CONTENT --------------------
with right_col:

    # ---------- HOME PAGE ----------
    if selected == 'Home':
        st.subheader("Project Introduction")
        st.write(
            "This project focuses on predicting medical insurance charges using Machine Learning. "
            "Healthcare costs vary significantly based on factors like age, BMI, smoking habits, and region. "
            "By leveraging data-driven approaches, this project aims to provide accurate cost predictions."
        )
        st.write(
            "The goal is to make healthcare cost estimation more transparent and accessible for both users and insurers."
        )

        st.markdown("---")
        st.markdown("### 📊 Key Insights from the Dataset")
        st.caption("Quick statistics summarizing policyholders’ demographics and insurance data:")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📏 Avg. BMI", round(data['bmi'].mean(), 2))
        col2.metric("💵 Avg. Charges", round(data['charges'].mean(), 2))
        col3.metric("👶 Avg. Children", int(round(data['children'].mean(), 0)))
        col4.metric("🎂 Avg. Age", int(round(data['age'].mean(), 0)))

        st.markdown("---")
        st.markdown("### 🧾 Raw Dataset Preview")
        st.caption("Snapshot of the dataset used for training and analysis:")
        st.dataframe(data1, use_container_width=True)

    

    # -------------------- ANALYSIS PAGE --------------------
    elif selected == 'Analysis':
        st.markdown("## 📊 Data Analysis")

        choices = option_menu(
        menu_title=None,
        options=['Univariate', 'Bivariate', 'Multivariate', 'Co-Relation'],
        icons=['bi-bar-chart-fill', 'bi-diagram-2', 'bi-grid-3x3', 'bi-arrow-down-up'],
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#0E1117", "justify-content": "center"},
            "icon": {"color": "#00B4D8", "font-size": "20px"},
            "nav-link": {
                "font-size": "17px",
                "text-align": "center",
                "margin": "0px 15px",
                "--hover-color": "#262730"
            },
            "nav-link-selected": {"background-color": "#00B4D8"},
        })
        # Univariate Analysis
        if choices == 'Univariate':
            with st.container(border = True):
                st.markdown("### 🔹 Univariate Analysis")
                st.write("Explore how individual variables are distributed across policyholders.")

            # ---- LEFT SIDE MENU ----
            col_menu, col_content = st.columns([1.2, 5], gap = 'large')
            with col_menu:
                with st.container(border = True):
                    selected_type = option_menu(
                    menu_title="Variable Type",
                    options=["Numerical", "Categorical"],
                    icons=["123", "list"],
                    menu_icon="filter-circle",
                    default_index=0,
                    orientation="vertical",
                    styles={
                        "container": {"padding": "0!important", "background-color": "#0E1117"},
                        "icon": {"color": "#00B4D8", "font-size": "18px"},
                        "nav-link": {
                            "font-size": "15px",
                            "text-align": "left",
                            "margin": "2px 0px",
                            "--hover-color": "#262730"
                        },
                        "nav-link-selected": {"background-color": "#00B4D8"},
                    }
                )

            # ---- RIGHT SIDE CONTENT ----
            with col_content:
                with st.container(border = True):
                # ============= NUMERICAL VARIABLES =============
                    if selected_type == "Numerical":
                        st.markdown("#### 📊 Numerical Features")

                        # Create tabs for Charges, BMI, and Age
                        tab1, tab2, tab3 = st.tabs(["💵 Charges", "📏 BMI", "🎂 Age"])

                        # ---------------- Charges Tab ----------------
                        with tab1:
                            col1, col2 = st.columns([2, 1], gap='large')
                            with col1:
                                fig1 = px.histogram(
                                    data,
                                    x="charges",
                                    nbins=40,
                                    color_discrete_sequence=["#00B4D8"],
                                    title="Distribution of Insurance Charges"
                                )
                                fig1.update_layout(
                                    xaxis_title="Charges ($)",
                                    yaxis_title="Count",
                                    template="plotly_dark"
                                )
                                st.plotly_chart(fig1, use_container_width=True)

                            with col2:
                                st.markdown("#### 💬 Insights")
                                st.write("""
                                - The **distribution of charges** is right-skewed — most people pay relatively low amounts.
                                - A small group (often **smokers** or with **high BMI**) pay much higher charges.
                                - Consider using `log(charges)` during modeling to reduce skew.
                                """)

                        # ---------------- BMI Tab ----------------
                        with tab2:
                            col1, col2 = st.columns([2, 1], gap='large')
                            with col1:
                                fig2 = px.histogram(
                                    data,
                                    x="bmi",
                                    nbins=30,
                                    color_discrete_sequence=["#36CFC9"],
                                    title="Distribution of BMI (Body Mass Index)"
                                )
                                fig2.update_layout(
                                    xaxis_title="BMI",
                                    yaxis_title="Count",
                                    template="plotly_dark"
                                )
                                st.plotly_chart(fig2, use_container_width=True)

                            with col2:
                                st.markdown("#### 💬 Insights")
                                st.write("""
                                - The **BMI distribution** is approximately normal with a slight right skew.
                                - High BMI (>30) often correlates with **higher insurance costs**.
                                - The healthy BMI range is **18.5–24.9**.
                                """)

                        # ---------------- Age Tab ----------------
                        with tab3:
                            col1, col2 = st.columns([2, 1], gap='large')
                            with col1:
                                fig3 = px.histogram(
                                    data,
                                    x="age",
                                    nbins=30,
                                    color_discrete_sequence=["#A066FF"],
                                    title="Distribution of Policyholder Age"
                                )
                                fig3.update_layout(
                                    xaxis_title="Age",
                                    yaxis_title="Count",
                                    template="plotly_dark"
                                )
                                st.plotly_chart(fig3, use_container_width=True)

                            with col2:
                                st.markdown("#### 💬 Insights")
                                st.write("""
                                - Most policyholders are between **20–50 years old**.
                                - Older individuals generally face **higher charges**.
                                - Age shows a **positive correlation** with cost.
                                """)

                # ==    =========== CATEGORICAL VARIABLES =============
                    elif selected_type == "Categorical":
                        st.markdown("#### 🧩 Categorical Features")

                        # Tabs for categorical variables
                        tab1, tab2, tab3, tab4 = st.tabs(["🚻 Gender", "🚬 Smoker", "🌍 Region", "👶 Number of Children"])

                        # ---------------- GENDER TAB ----------------
                        with tab1:
                            col1, col2 = st.columns([2, 1], gap='large')
                            with col1:
                                gender_counts = data1['sex'].value_counts()
                                fig4 = px.pie(
                                    values=gender_counts.values,
                                    names=gender_counts.index,
                                    color_discrete_sequence=px.colors.sequential.Tealgrn,
                                    title="Gender Distribution"
                                )
                                st.plotly_chart(fig4, use_container_width=True)

                            with col2:
                                st.markdown("#### 💬 Insights")
                                st.write("""
                                - Gender distribution appears roughly balanced between **male** and **female**.
                                - Although gender has some effect on charges, it’s usually less significant compared to **smoking** or **BMI**.
                                """)

                        # ---------------- SMOKER TAB ----------------
                        with tab2:
                            col1, col2 = st.columns([2, 1], gap='large')
                            with col1:
                                smoker_counts = data1['smoker'].value_counts()
                                fig5 = px.pie(
                                    values=smoker_counts.values,
                                    names=smoker_counts.index,
                                    color_discrete_sequence=px.colors.sequential.RdBu,
                                    title="Smoker vs Non-Smoker"
                                )
                                st.plotly_chart(fig5, use_container_width=True)

                            with col2:
                                st.markdown("#### 💬 Insights")
                                st.write("""
                                - Smokers make up a smaller portion of the dataset, yet pay **significantly higher charges**.
                                - This variable is among the **strongest predictors** of insurance cost.
                                """)

                            # ---------------- REGION TAB ----------------
                            with tab3:
                                col1, col2 = st.columns([2, 1], gap='large')
                                with col1:
                                    region_counts = data1['region'].value_counts()
                                    fig6 = px.bar(
                                        x=region_counts.index,
                                        y=region_counts.values,
                                        color=region_counts.index,
                                        color_discrete_sequence=px.colors.sequential.Tealgrn,
                                        title="Regional Distribution of Policyholders"
                                    )
                                    fig6.update_layout(xaxis_title="Region", yaxis_title="Count", template="plotly_dark")
                                    st.plotly_chart(fig6, use_container_width=True)

                                with col2:
                                    st.markdown("#### 💬 Insights")
                                    st.write("""
                                    - Most policyholders belong to the **southeast** region.
                                    - Geographic differences can reflect variations in **healthcare costs** or **income levels**.
                                    """)

                            # ---------------- CHILDREN TAB ----------------
                            with tab4:
                                col1, col2 = st.columns([2, 1], gap='large')
                                with col1:
                                    children_counts = data1['children'].value_counts().sort_index()
                                    fig7 = px.bar(
                                        x=children_counts.index,
                                        y=children_counts.values,
                                        color=children_counts.index,
                                        color_discrete_sequence=px.colors.sequential.RdPu,
                                        title="Number of Children per Policyholder"
                                    )
                                    fig7.update_layout(xaxis_title="Number of Children", yaxis_title="Count", template="plotly_dark")
                                    st.plotly_chart(fig7, use_container_width=True)

                                with col2:
                                    st.markdown("#### 💬 Insights")
                                    st.write("""
                                    - Most policyholders have **0 to 2 children**.
                                    - The number of dependents can slightly increase insurance costs, but the relationship is **not very strong**.
                                    """)

        elif choices == 'Bivariate':
            with st.container(border = True):
                st.markdown("### 🔹 Bivariate Analysis")
                st.markdown("Relationship Between Insurance Charges and Other Factors")
                st.write("Use the filters below to explore how **Age** and **BMI** affect medical insurance charges interactively.")

            # ---- MAIN LAYOUT ----
            col_menu, col_content = st.columns([4, 9], gap='large')

            # ---- LEFT SIDE MENU + FILTERS ----
            with col_menu:
                with st.container(border=True):
                    selected_type = option_menu(
                        menu_title="Variable Type",
                        options=["Numerical", "Categorical"],
                        icons=["123", "list"],
                        menu_icon="filter-circle",
                        default_index=0,
                        orientation="vertical",
                        styles={
                            "container": {"padding": "0!important", "background-color": "#0E1117"},
                            "icon": {"color": "#00B4D8", "font-size": "18px"},
                            "nav-link": {
                                "font-size": "15px",
                                "text-align": "left",
                                "margin": "2px 0px",
                                "--hover-color": "#262730"
                            },
                            "nav-link-selected": {"background-color": "#00B4D8"},
                        }
                    )

                # --- FILTER SECTION BELOW MENU ---

                with st.container(border=True):

                        min_charge, max_charge = int(data['charges'].min()), int(data['charges'].max())
                        charge_range = st.slider("Select Charges Range ($):", min_charge, max_charge, (min_charge, max_charge))
                        st.markdown("<br>", unsafe_allow_html=True)

                        if selected_type == 'Numerical': 
                            min_age, max_age = int(data['age'].min()), int(data['age'].max())
                            age_range = st.slider("Select Age Range:", min_age, max_age, (min_age, max_age))
                            st.markdown("<br>", unsafe_allow_html=True)

                        if selected_type == 'Numerical':
                            min_bmi, max_bmi = float(data['bmi'].min()), float(data['bmi'].max())
                            bmi_range = st.slider("Select BMI Range:", min_bmi, max_bmi, (min_bmi, max_bmi))
                            st.markdown("<br>", unsafe_allow_html=True)

                        if selected_type == 'Numerical':
                            # Apply Filters
                            filtered_df = data[
                            (data['charges'].between(*charge_range)) &
                            (data['age'].between(*age_range)) &
                            (data['bmi'].between(*bmi_range))
                            ]

                        if selected_type == 'Categorical':
                            # Filter by Smoker
                            smoker_options = data['smoker'].unique().tolist()
                            selected_smokers = st.multiselect("Select Smoker Type:", smoker_options, default=smoker_options)
                            st.markdown("<br>", unsafe_allow_html=True)

                        if selected_type == 'Categorical':
                            # Filter by Sex
                            sex_options = data['sex'].unique().tolist()
                            selected_sex = st.multiselect("Select Gender:", sex_options, default=sex_options)
                            st.markdown("<br>", unsafe_allow_html=True)

                        if selected_type == 'Categorical':
                            # Filter by Number of Children
                            child_options = sorted(data['children'].unique().tolist())
                            selected_children = st.multiselect("Select Number of Children:", child_options, default=child_options)
                            st.markdown("<br>", unsafe_allow_html=True)

                        if selected_type == 'Categorical':
                            # Apply filters
                            filtered_df = data[
                                (data['charges'].between(*charge_range)) &
                                (data['smoker'].isin(selected_smokers)) &
                                (data['sex'].isin(selected_sex)) &
                                (data['children'].isin(selected_children))
                            ]

            # ---- RIGHT SIDE CONTENT ----
            with col_content:
                with st.container(border=True):
                    # ============= NUMERICAL VARIABLES =============
                    if selected_type == "Numerical":
                        st.markdown("#### 📊 Numerical Features")

                        # --- Tabs for Scatter Plots ---
                        tab1, tab2 = st.tabs(["🎂 Charges vs Age", "📏 Charges vs BMI"])

                        # ---------------- TAB 1: Charges vs AGE ----------------
                        with tab1:
                            col1, col2 = st.columns([2, 1], gap="large")

                            with col1:
                                fig_age = px.scatter(
                                    filtered_df,
                                    x="age",
                                    y="charges",
                                    color="smoker",
                                    size="bmi",
                                    hover_data=["sex", "region"],
                                    color_discrete_sequence=px.colors.qualitative.Set1,
                                    title="Charges vs Age"
                                )
                                fig_age.update_layout(
                                    xaxis_title="Age",
                                    yaxis_title="Charges ($)",
                                    template="plotly_dark"
                                )
                                st.plotly_chart(fig_age, use_container_width=True)

                            with col2:
                                st.markdown("#### 💬 Insights")
                                st.write("""
                                - As **age increases**, insurance charges generally rise — reflecting higher health risk with age.  
                                - **Smokers** consistently appear in the higher charge range across all ages.  
                                - **BMI size** (represented by point size) also shows that higher BMI may increase charges.
                                """)

                        # ---------------- TAB 2: Charges vs BMI ----------------
                        with tab2:
                            col1, col2 = st.columns([2, 1], gap="large")

                            with col1:
                                fig_bmi = px.scatter(
                                    filtered_df,
                                    x="bmi",
                                    y="charges",
                                    color="smoker",
                                    size="age",
                                    hover_data=["sex", "region"],
                                    color_discrete_sequence=px.colors.qualitative.Safe,
                                    title="Charges vs BMI"
                                )
                                fig_bmi.update_layout(
                                    xaxis_title="BMI",
                                    yaxis_title="Charges ($)",
                                    template="plotly_dark"
                                )
                                st.plotly_chart(fig_bmi, use_container_width=True)

                            with col2:
                                st.markdown("#### 💬 Insights")
                                st.write("""
                                - Charges increase significantly beyond a **BMI of 30 (Obese range)**.  
                                - Smokers with high BMI values incur **much higher costs** than non-smokers.  
                                - Younger individuals (smaller points) generally have lower charges, regardless of BMI.
                                """)

                    # ============= CATEGORICAL VARIABLES =============
                    elif selected_type == "Categorical":
                        st.markdown("#### 🧩 Categorical Features")

                        # ==============================
                        # 📊 Tabs for Each Category
                        # ==============================
                        tab1, tab2, tab3 = st.tabs(["🚬 Smoker", "🚻 Gender", "👶 Children"])

                        # ---- SMOKER TAB ----
                        with tab1:
                            col1, col2 = st.columns([2, 1], gap='large')

                            with col1:
                                fig1 = px.box(
                                    filtered_df,
                                    x="smoker",
                                    y="charges",
                                    color="smoker",
                                    color_discrete_sequence=px.colors.sequential.RdBu,
                                    title="Charges Distribution by Smoking Status"
                                )
                                fig1.update_layout(
                                    xaxis_title="Smoker",
                                    yaxis_title="Charges ($)",
                                    template="plotly_dark"
                                )
                                st.plotly_chart(fig1, use_container_width=True)

                            with col2:
                                st.markdown("#### 💬 Insights")
                                st.write("""
                                - **Smokers** pay significantly higher medical charges than non-smokers.
                                - The distribution is much more spread out for smokers, indicating high variance.
                                - Smoking status is a **key factor** in cost prediction.
                                """)

                        # ---- GENDER TAB ----
                        with tab2:
                            col1, col2 = st.columns([2, 1], gap='large')

                            with col1:
                                fig2 = px.box(
                                    filtered_df,
                                    x="sex",
                                    y="charges",
                                    color="sex",
                                    color_discrete_sequence=px.colors.sequential.Tealgrn,
                                    title="Charges Distribution by Gender"
                                )
                                fig2.update_layout(
                                    xaxis_title="Gender",
                                    yaxis_title="Charges ($)",
                                    template="plotly_dark"
                                )
                                st.plotly_chart(fig2, use_container_width=True)

                            with col2:
                                st.markdown("#### 💬 Insights")
                                st.write("""
                                - The difference between **male** and **female** charges is relatively small.
                                - Gender alone is **not a strong predictor** of insurance cost.
                                - It may interact with other variables (like BMI or smoker status).
                                """)

                        # ---- CHILDREN TAB ----
                        with tab3:
                            col1, col2 = st.columns([2, 1], gap='large')

                            with col1:
                                fig3 = px.box(
                                    filtered_df,
                                    x="children",
                                    y="charges",
                                    color="children",
                                    color_discrete_sequence=px.colors.sequential.Purpor,
                                    title="Charges Distribution by Number of Children"
                                )
                                fig3.update_layout(
                                    xaxis_title="Number of Children",
                                    yaxis_title="Charges ($)",
                                    template="plotly_dark"
                                )
                                st.plotly_chart(fig3, use_container_width=True)

                            with col2:
                                st.markdown("#### 💬 Insights")
                                st.write("""
                                - There’s **no strong trend** between number of children and charges.
                                - Having more dependents may slightly increase total charges due to coverage, but not strongly.
                                - Variability is driven more by **smoking** and **BMI** than dependents.
                                """)

        if choices == 'Multivariate':
            with st.container(border=True):
                st.markdown("### 🔹 Multivariate Analysis")
                st.write("Explore how multiple variables jointly influence insurance charges.")

            col_menu, col_content = st.columns([3, 9], gap='large')

            # ---- LEFT SIDE MENU + FILTERS ----
            with col_menu:
                with st.container(border=True):
                    selected_type = option_menu(
                        menu_title="Pattern Type",
                        options=["Smoker-Based Patterns", "BMI-related Patterns"],
                        icons=["fire", "activity"],
                        menu_icon="filter-circle",
                        default_index=0,
                        orientation="vertical",
                        styles={
                            "container": {"padding": "0!important", "background-color": "#0E1117"},
                            "icon": {"color": "#00B4D8", "font-size": "18px"},
                            "nav-link": {"font-size": "15px", "text-align": "left", "margin": "2px 0px"},
                            "nav-link-selected": {"background-color": "#00B4D8"},
                        }
                    )

                with st.container(border=True):
                    min_charge, max_charge = int(data['charges'].min()), int(data['charges'].max())
                    charge_range = st.slider("Select Charges Range ($):", min_charge, max_charge, (min_charge, max_charge))
                    st.markdown("<br>", unsafe_allow_html=True)

                    smoker_options = data['smoker'].unique().tolist()
                    selected_smokers = st.multiselect("Select Smoker Type:", smoker_options, default=smoker_options)

                    sex_options = data['sex'].unique().tolist()
                    selected_sex = st.multiselect("Select Gender:", sex_options, default=sex_options)

                    region_options = data['region'].unique().tolist()
                    selected_region = st.multiselect("Select Region:", region_options, default=region_options)

                    min_age, max_age = int(data['age'].min()), int(data['age'].max())
                    age_range = st.slider("Select Age Range:", min_age, max_age, (min_age, max_age))

                    min_bmi, max_bmi = float(data['bmi'].min()), float(data['bmi'].max())
                    bmi_range = st.slider("Select BMI Range:", min_bmi, max_bmi, (min_bmi, max_bmi))

                    # --- Apply filters safely ---
                    filtered_df = data[
                        (data['charges'].between(*charge_range)) &
                        (data['smoker'].isin(selected_smokers)) &
                        (data['sex'].isin(selected_sex)) &
                        (data['region'].isin(selected_region)) &
                        (data['age'].between(*age_range)) &
                        (data['bmi'].between(*bmi_range))
                    ]

            # ---- RIGHT SIDE CONTENT ----
            with col_content:
                with st.container(border=True):
                    if selected_type == "Smoker-Based Patterns":
                        st.markdown("#### 📊 Smoker-Based Patterns")

                        tab1, tab2 = st.tabs(["🚻 Gender vs Charges", "🌍 Region vs Charges"])

                        # TAB 1: GENDER
                        with tab1:
                            col1, col2 = st.columns([2, 1], gap="large")
                            with col1:
                                fig1 = px.box(
                                    filtered_df,
                                    x="sex",
                                    y="charges",
                                    color="sex",
                                    title="Charges by Gender (Smokers Only)",
                                    template="plotly_dark"
                                )
                                st.plotly_chart(fig1, use_container_width=True)

                            with col2:
                                st.markdown("#### 💬 Insights")
                                st.write("""
                                - Both male and female **smokers** show high insurance charges.  
                                - Gender-based difference is small but visible at higher ranges.
                                """)

                        # TAB 2: REGION
                        with tab2:
                            col1, col2 = st.columns([2, 1], gap="large")
                            with col1:
                                fig2 = px.box(
                                    filtered_df,
                                    x="region",
                                    y="charges",
                                    color="region",
                                    title="Charges by Region (Smokers Only)",
                                    template="plotly_dark"
                                )
                                st.plotly_chart(fig2, use_container_width=True)

                            with col2:
                                st.markdown("#### 💬 Insights")
                                st.write("""
                                - **Southeast region** shows relatively higher median charges for smokers.  
                                - Regional variation could be due to **income** and **healthcare access** differences.
                                """)

                    # ---- BMI-RELATED PATTERNS ----
                    elif selected_type == "BMI-related Patterns":
                        st.markdown("#### ⚖️ BMI-Related Patterns")

                        tab1, tab2 = st.tabs(["💨 Charges vs Age, BMI, and Smoker", "🏋️ Obese vs Non-Obese Comparison"])

                        # TAB 1: SCATTER
                        with tab1:
                            col1, col2 = st.columns([2, 1], gap='large')
                            with col1:
                                fig1 = px.scatter(
                                    filtered_df,
                                    x="age",
                                    y="charges",
                                    color="smoker",
                                    size="bmi",
                                    hover_data=["sex", "region", "children"],
                                    color_discrete_map={"yes": "#FF595E", "no": "#1982C4"},
                                    title="Charges vs Age, BMI, and Smoker Status",
                                    template="plotly_dark"
                                )
                                st.plotly_chart(fig1, use_container_width=True)

                            with col2:
                                st.markdown("#### 💬 Insights")
                                st.write("""
                                - **Smokers** show consistently higher charges across all ages.  
                                - Higher **BMI** intensifies cost, especially among smokers.  
                                - Younger non-smokers cluster at lower charge ranges.
                                """)

                        # TAB 2: BAR COMPARISON
                        with tab2:
                            col1, col2 = st.columns([2, 1], gap='large')
                            obese_smokers = filtered_df[(filtered_df['bmi'] > 30) & (filtered_df['smoker'] == 'yes')]
                            non_obese_non_smokers = filtered_df[(filtered_df['bmi'] <= 30) & (filtered_df['smoker'] == 'no')]

                            avg_charges = pd.DataFrame({
                                "Group": ["Obese Smokers", "Non-Obese Non-Smokers"],
                                "Average Charges": [
                                    obese_smokers['charges'].mean(),
                                    non_obese_non_smokers['charges'].mean()
                                ]
                            })

                            with col1:
                                fig2 = px.bar(
                                    avg_charges,
                                    x="Group",
                                    y="Average Charges",
                                    color="Group",
                                    color_discrete_sequence=["#FF595E", "#1982C4"],
                                    title="Average Charges Comparison",
                                    template="plotly_dark"
                                )
                                st.plotly_chart(fig2, use_container_width=True)

                            with col2:
                                st.markdown("#### 💬 Insights")
                                st.write("""
                                - **Obese smokers** pay **~2x higher** insurance on average.  
                                - Smoking and obesity have a **compounding effect** on charges.
                                """)

                        st.markdown("---")
                        st.info("💡 Tip: Use filters to isolate specific demographics or regions for detailed insight.")
        elif choices == 'Co-Relation':
            with st.container(border=True):
                st.markdown("### 🔹 Correlation Analysis")
                st.write("Explore how numerical variables relate to one another in the dataset.")

            # ---- Layout ----
            col_menu, col_content = st.columns([3, 9], gap='large')

            # ---------------- LEFT SIDE FILTERS ----------------
            with col_menu:
                with st.container(border=True):
                    st.markdown("#### 🔧 Filter the Data")

                    # Filters for numerical ranges
                    min_age, max_age = int(data['age'].min()), int(data['age'].max())
                    age_range = st.slider("Select Age Range:", min_age, max_age, (min_age, max_age))

                    min_bmi, max_bmi = float(data['bmi'].min()), float(data['bmi'].max())
                    bmi_range = st.slider("Select BMI Range:", min_bmi, max_bmi, (min_bmi, max_bmi))

                    min_children, max_children = int(data['children'].min()), int(data['children'].max())
                    children_range = st.slider("Select Children Range:", min_children, max_children, (min_children, max_children))

                    min_charge, max_charge = int(data['charges'].min()), int(data['charges'].max())
                    charge_range = st.slider("Select Charges Range ($):", min_charge, max_charge, (min_charge, max_charge))

                    # Apply filters
                    filtered_df = data[
                        (data['age'].between(*age_range)) &
                        (data['bmi'].between(*bmi_range)) &
                        (data['children'].between(*children_range)) &
                        (data['charges'].between(*charge_range))
                    ]

            # ---------------- RIGHT SIDE: HEATMAP + INSIGHTS ----------------
            with col_content:
                with st.container(border=True):
                    st.markdown("#### 📈 Correlation Heatmap")

                    # Select numerical columns
                    numerical_features = ['age', 'bmi', 'children', 'charges']
                    corr_matrix = filtered_df[numerical_features].corr()

                    # Create Plotly heatmap
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        color_continuous_scale="RdBu_r",
                        aspect="auto",
                        title="Correlation Matrix of Numerical Features",
                        template="plotly_dark"
                    )

                    col1, col2 = st.columns([2, 1], gap='large')
                    with col1:
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.markdown("#### 💬 Insights")
                        st.write("""
                        - **Charges** show the **strongest positive correlation** with both **age** and **BMI**.  
                        - **Children** has a weak correlation with charges, meaning family size doesn’t majorly impact cost.  
                        - The correlation between **age and BMI** is near zero, implying they vary independently.  
                        - Stronger correlations (closer to ±1) indicate **predictive potential** for modeling.
                        """)

                st.markdown("---")
                st.info("💡 Tip: Adjust filters on the left to see how correlations shift across demographics or price ranges.")            




            # Other analysis sections (Bivariate, Multivariate, Outlier Analysis)
            # keep your full logic here as it was — I just fixed the layout and menu structure

    # -------------------- PREDICTION PAGE --------------------
    elif selected == 'Prediction':
         # ========== Load Model and Scaler ==========
         @st.cache_resource
         def load_model_and_scaler():
             model = joblib.load("models/gradient_boosting_model_updated.joblib")
             scaler = joblib.load("models/robust_scaler.joblib")
             return model, scaler

         model, scaler = load_model_and_scaler()

         # ========== Page Title ==========
         st.markdown("## 💡 Insurance Charges Prediction")
         st.write("Enter your details on the left to get an estimate of your **medical insurance cost**.")

         # === Two-column layout (form left, results right) ===
         input_col, output_col = st.columns([2, 2], gap="large")

         with input_col:
             st.markdown("### 🧾 Input Details")
             with st.form("prediction_form"):
                 col1, col2 = st.columns(2)
                 with col1:
                     age = st.slider("Age", 18, 80, 30)
                     sex = st.selectbox("Sex", ["male", "female"])
                     bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
                 with col2:
                     children = st.number_input("Number of Children", min_value=0, max_value=5, value=0, step=1)
                     smoker = st.selectbox("Smoker", ["yes", "no"])
                     region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
                 submitted = st.form_submit_button("🔮 Predict")

         with output_col:
             if submitted:
                 # Prepare input for model
                 input_dict = {
                     'age': age,
                     'bmi': bmi,
                     'children': children,
                     'smoker': 1 if smoker == 'yes' else 0,
                     'sex_male': 1 if sex == 'male' else 0,
                     'region_northwest': 1 if region == 'northwest' else 0,
                     'region_southeast': 1 if region == 'southeast' else 0,
                     'region_southwest': 1 if region == 'southwest' else 0
                 }
                 input_data = pd.DataFrame([input_dict])

                 # Scale numeric columns
                 numeric_cols = ['age', 'bmi', 'children']
                 input_data_scaled = input_data.copy()
                 input_data_scaled[numeric_cols] = scaler.transform(input_data[numeric_cols])

                 # Ensure correct order
                 input_data_scaled = input_data_scaled.reindex(columns=[
                     'age', 'bmi', 'children', 'smoker', 'sex_male',
                     'region_northwest', 'region_southeast', 'region_southwest'
                 ])

                 # Prediction
                 prediction = model.predict(input_data_scaled)[0]

                 # === Tabs appear in the right column ===
                 tab1, tab2 = st.tabs(["Prediction & Explanation", "Confidence Interval"])

                 # ------------------ TAB 1 ------------------
                 with tab1:
                     st.success(f"💰 **Predicted Insurance Charges:** ${prediction:,.2f}")
                     explanation = []

                     if smoker == 'yes':
                         explanation.append("🚬 Being a smoker significantly increases insurance charges.")
                     else:
                         explanation.append("✅ Non-smokers usually have much lower insurance costs.")

                     if bmi > 30:
                         explanation.append("⚠️ High BMI (>30) suggests higher health risks, raising costs.")
                     elif bmi < 18.5:
                         explanation.append("⚖️ Low BMI can reduce costs but may have other risks.")
                     else:
                         explanation.append("💪 Healthy BMI keeps costs moderate.")

                     if age > 50:
                         explanation.append("📈 Older individuals have higher medical costs.")
                     else:
                         explanation.append("🧒 Younger individuals tend to have lower costs.")

                     explanation.append(f"🌍 Region: living in **{region}** may slightly affect prices.")

                     st.markdown("### 🤔 Why this prediction?")
                     for point in explanation:
                         st.markdown(f"- {point}")

                     # SHAP Explanation
                     import shap
                     import matplotlib.pyplot as plt

                     st.markdown("### 📊 SHAP Explanation (Feature Impact)")
                     explainer = shap.Explainer(model)
                     shap_values = explainer(input_data_scaled)

                     fig, ax = plt.subplots()
                     shap.plots.waterfall(shap_values[0], show=False)
                     with st.expander("Show SHAP Waterfall Plot"):
                         st.pyplot(fig)

                 # ------------------ TAB 2 ------------------
                 with tab2:
                     import numpy as np
                     import matplotlib.pyplot as plt

                     preds = []
                     n_bootstraps = 200
                     rng = np.random.default_rng(42)
                     for i in range(n_bootstraps):
                         noise = rng.normal(0, 0.01, size=input_data_scaled.shape)
                         boot_input = input_data_scaled + noise
                         preds.append(model.predict(boot_input)[0])

                     lower_bound = np.percentile(preds, 2.5)
                     upper_bound = np.percentile(preds, 97.5)

                     st.success(f"💰 **Predicted Insurance Charges:** ${prediction:,.2f}")
                     st.info(f"🔍 95% Confidence Interval: **${lower_bound:,.2f} – ${upper_bound:,.2f}**")

                     fig, ax = plt.subplots()
                     ax.hist(preds, bins=20)
                     ax.axvline(prediction, color='red', linestyle='--', label='Predicted')
                     ax.axvline(lower_bound, color='green', linestyle='--', label='95% CI Lower')
                     ax.axvline(upper_bound, color='green', linestyle='--', label='95% CI Upper')
                     ax.set_title("Bootstrap Distribution of Predictions")
                     ax.legend()
                     st.pyplot(fig)

                     st.markdown("""
                     ---
                     ### 🧠 Interpretation:
                     - **Red dashed line:** Model’s predicted cost  
                     - **Green dashed lines:** 95% confidence limits  
                     - **Narrow interval →** high certainty  
                     - **Wide interval →** more uncertainty  
                     """)


    # -------------------- MODEL PERFORMANCE PAGE --------------------
    elif selected == "Model Performances":
        st.markdown("## 📊 Model Performance Comparison")
        st.write("Compare different regression models based on key evaluation metrics.")

        # ================== DATA ==================
        df = pd.DataFrame({
            "Model": ["Linear Regression", "Random Forest", "XGBoost", "Ridge", "Gradient Boosting"],
            "RMSE": [5956.342894, 4701.888728, 4978.712288, 5972.641624, 4268.257507],
            "MAE": [4177.045561, 2639.951110, 2922.593714, 4193.954998, 2517.373211],
            "R-squared": [0.806929, 0.879690, 0.865106, 0.805871, 0.900858],
        })

        # ================== OPTION MENU ==================
        selected_metric = option_menu(
            menu_title="Select a Performance Metric",
            options=["RMSE", "MAE", "R-squared"],
            icons=["activity", "bar-chart", "graph-up-arrow"],
            orientation="horizontal",
            default_index=0
        )

        st.markdown(f"### 📈 Model Comparison — **{selected_metric}**")

        # ================== DETERMINE BEST MODEL ==================
        if selected_metric in ["RMSE", "MAE"]:
            best_model = df.loc[df[selected_metric].idxmin(), "Model"]
        else:
            best_model = df.loc[df[selected_metric].idxmax(), "Model"]

        # ================== PLOTLY BAR CHART ==================
        color_col = ["#636EFA" if model != best_model else "#EF553B" for model in df["Model"]]

        fig = px.bar(
            df,
            x="Model",
            y=selected_metric,
            text=df[selected_metric].apply(lambda x: f"{x:,.2f}"),
            color=df["Model"],
            color_discrete_sequence=color_col,
        )

        fig.update_traces(textposition='outside')
        fig.update_layout(
            title=f"{selected_metric} Comparison Across Models",
            xaxis_title="Model",
            yaxis_title=selected_metric,
            showlegend=False,
            height=500,
            template="plotly_white"
        )

        if selected_metric in ["RMSE", "MAE"]:
            fig.update_yaxes(autorange="reversed")  # Lower is better

        st.plotly_chart(fig, use_container_width=True)

        # ================== METRIC EXPLANATION ==================
        st.markdown("---")
        st.markdown(f"### 🧠 Understanding **{selected_metric}**")

        if selected_metric == "RMSE":
            st.write("""
            **Root Mean Squared Error (RMSE)** measures the square root of the average squared difference between predicted and actual values.
            - It penalizes **large errors** more heavily than smaller ones.  
            - **Lower RMSE → Better model fit.**
            """)
        elif selected_metric == "MAE":
            st.write("""
            **Mean Absolute Error (MAE)** represents the average of the absolute differences between predicted and true values.  
            - It provides a **straightforward measure of average error magnitude**.  
            - **Lower MAE → More accurate predictions.**
            """)
        else:
            st.write("""
            **R-squared (R²)** indicates how well the model explains variance in the target variable.  
            - **Ranges from 0 to 1**, with higher values meaning better explanatory power.  
            - **Higher R² → Better model fit.**
            """)

        # ================== INTERPRETATION ==================
        st.markdown("### 📊 Interpretation of Comparison Plot")
        if selected_metric in ["RMSE", "MAE"]:
            st.info(f"""
            - **Shorter bars** indicate better performance since lower error = higher accuracy.  
            - The best-performing model based on **{selected_metric}** is **{best_model}**.  
            - This model predicts insurance costs **more precisely** than others.
            """)
        else:
            st.info(f"""
            - **Taller bars** indicate better performance since higher R² = stronger explanatory power.  
            - The model with the highest R² is **{best_model}**, meaning it explains the most variance in insurance charges.  
            - This model best captures the relationships in your data.
            """)

        # ================== SUMMARY TABLE ==================
        st.markdown("---")
        st.markdown("### 🧾 Full Model Comparison Table")
        st.dataframe(
            df.style.format({
                "RMSE": "{:,.2f}",
                "MAE": "{:,.2f}",
                "R-squared": "{:.4f}"
            }).highlight_max(subset=["R-squared"], color="#90EE90").highlight_min(subset=["RMSE", "MAE"], color="#FFB6C1")
        )
