import streamlit as st
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
# Setup the Page
jadeimage = Image.open("assets/jadeglobalsmall.png")
st.set_page_config(page_title="Jade Global", page_icon=jadeimage, layout="wide")
st_UserName = st.secrets["streamlit_username"]
st_Password = st.secrets["streamlit_password"]

st_user = st_UserName.upper()

def creds_entered():
    if len(st.session_state["streamlit_username"]) > 0 and len(st.session_state["streamlit_password"]) > 0:
        if st.session_state["streamlit_username"].strip() == st_UserName \
                and st.session_state["streamlit_password"].strip() == st_Password:
            st.session_state["authenticated"] = True
        else:
            st.session_state["authenticated"] = False
            st.error("Invalid Username/Password ")


def authenticate_user():
    if "authenticated" not in st.session_state:
        buff, col, buff2 = st.columns([1, 1, 1])
        col.text_input(label="Username", value="", key="streamlit_username", on_change=creds_entered)
        col.text_input(label="Password", value="", key="streamlit_password", type="password", on_change=creds_entered)
        return False
    else:
        if st.session_state["authenticated"]:
            return True
        else:
            buff, col, buff2 = st.columns([1, 1, 1])
            col.text_input(label="Username:", value="", key="streamlit_username", on_change=creds_entered)
            col.text_input(label="Password:", value="", key="streamlit_password", type="password",
                           on_change=creds_entered)
            return False
# A simple login check function
def check_login(username, password):
    # You can replace this with a more secure authentication method
    if username == "jade" and password == "Jade@123":
        return True
    return False

# Page 1 content
def page_1():
    #st.set_page_config(
    #    page_title="Yield Prediction",
    #    page_icon="🏚️", layout="wide"
    #)
    
    st.markdown(f"""
        <style>
            .barheader {{
                position: fixed;
                top: 0;
                left: 0;
                /*width: 244px;*/
                width: 20%;
                background-color: #175388;
                color: white;
                height: 50px;
                z-index: 1000;
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px 20px;
                text-align: center;
                font-size: 27px;
                font-weight: bold;
                margin: 0;
                //border-radius: 8px 8px 0 0;
            }}                        
        </style>
        <div class = "barheader"></div>
        """,unsafe_allow_html=True)

    st.subheader(':blue[Yield Monitoring Solution Objective]')
    #with st.sidebar:
    #    st.sidebar.success("Select a page above.")
    #    st.image('assets/jadeglobalbig.png')
    st.markdown(
        """
        Semiconductor manufacturing involves intricate processes where a single silicon wafer contains multiple microelectronic integrated circuit units, known as dies.
        
        Each wafer undergoes numerous steps in the production line, with advanced tools and techniques employed to create high-precision microchips. 
        
        However, various challenges in the production process, such as tool failures, mismatched operations, and unintended effects from one process step on another, can introduce variability into the system, ultimately impacting the yield and quality of the final product.
        
        In this case study, we explore how applying modern data analytics can significantly improve the yield of semiconductor manufacturing, thereby reducing costs and increasing profitability.
    """
    )

# Page 2 content
def page_2():
    import time

    import streamlit as st
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn import linear_model
    from sklearn import tree
    from sklearn.tree import DecisionTreeClassifier

    import streamlit as st
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from xgboost import XGBClassifier
    from sklearn.datasets import make_classification

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import LabelEncoder
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
    from sklearn.model_selection import RandomizedSearchCV, train_test_split
    from scipy.stats import randint
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, roc_auc_score
    from streamlit_extras.stylable_container import stylable_container

    #st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

    ##########################################################
    def highlight_odd_even(row):
        #return ['background-color: #E0FFFF' if row.name % 2 == 0 else 'background-color: #D3D3D3'] * len(row)
        return ['background-color: #e4f6ff; color: #4169E1' if row.name % 2 == 0 else 'background-color: #D3D3D3; color: #4169E1'] * len(row)
    ##########################################################    
    def style_rows(s):
        #return ['background-color: #E0FFFF' if row.name % 2 == 0 else 'background-color: #D3D3D3'] * len(row)
        return ['background-color: #e4f6ff; color: #4169E1' if s.name % 2 == 0 else 'background-color: #D3D3D3; color: #4169E1' for _ in s]
    ##########################################################    
    def highlight_header(col):
        return ['color: #E0FFFF'] * 1
    ##########################################################
    def highlight_data(s):
        return ['color: #4169E1' if v is None else 'color: #4169E1' for v in s]
    ##########################################################    
    def style_dataframe(df):
        return (df.style
                 .apply(style_rows, axis=1)
                 .set_table_styles({
                     '': {'selector': 'th', 'props': [('font-weight', 'bold'), ('color', 'coral')]}
                 }))
    ##########################################################    
    # Function to apply bold style to the header
    def bold_header(s):
        return ['font-weight: bold; color: coral' for _ in s]

        ########################### Declare functions for visualizations ############################################

    def color_negative_red(val):
        color = '#4169E1' if ((type(val) == int and val < 0) or str(val).find('-')) else '#FF0000'
        background = '#e4f6ff' if ((type(val) == int and val < 0) or str(val).find('-')) else '#D3D3D3'
        return f'color: {color}; background-color: {background}'    

    # Function to take the query as input and write the results to streamlit for TAM Maintenance ################
    def get_data(query,title_text):
        SF_Query_Usage_Data=session.sql(query)
        SF_Query_Usage_Data.collect()
        DF_SF_Query_Usage_Data = SF_Query_Usage_Data.to_pandas()  
        styled_df = DF_SF_Query_Usage_Data.style.apply(style_rows, axis=1).apply(highlight_data,axis=1).set_table_styles([{"selector": "th", "props": "color: blue;"}])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    ##########################################################
    def get_data_col1by3(query,title_text):
        SF_Query_Usage_Data=session.sql(query)
        SF_Query_Usage_Data.collect()
        DF_SF_Query_Usage_Data = SF_Query_Usage_Data.to_pandas()  
        styled_df = DF_SF_Query_Usage_Data.style.apply(style_rows, axis=1).apply(highlight_data,axis=1).set_table_styles([{"selector": "th", "props": "color: blue;"}])
        col1,col2,col3=st.columns(3)
        with col1:
            st.dataframe(styled_df, use_container_width=True, hide_index=True) 

    #with st.sidebar:
    #    st.image('assets/jadeglobalbig.png')
        
    st.subheader(':blue[Historical Data Analysis]')
    #elif option == "Dashboard":
    #st.header("Historical Data Analysis")

    st.session_state.data = pd.read_csv('Flat_Files/uci-secom.csv')
    #st.session_state.data = data
    # Display the DataFrame (showing the first 5 rows)
    #st.subheader("Data Preview:")
    #st.write(data.head())
    st.write("Data Preview:")
    df = st.session_state.data
    n=5
    df = df.groupby('wafer_id').apply(lambda group: group.head(n)).reset_index(drop = True)
    df = df.head(30)
    styled_df = df.style.apply(style_rows, axis=1).apply(highlight_data,axis=1).set_table_styles([{"selector": "th", "props": "color: blue;"}])
    st.dataframe(styled_df, use_container_width=True, hide_index=True) 

    # Display DataFrame shape (rows and columns)
    st.write(f"Number of rows: {st.session_state.data.shape[0]}")
    st.write(f"Number of columns: {st.session_state.data.shape[1]}")
    st.write(f"Number of wafer: {st.session_state.data.wafer_id.nunique()}")

    #st.title("Overall Pass/Fail Distribution")
    st.subheader(':blue[Overall Pass/Fail Distribution]')
    col1, col2 = st.columns(2)

    # Plot Confusion Matrix in the first column
    with col1:
        col1, col2 = st.columns(2)
        with col1:
            # Pie chart
            labels = ['Pass', 'Fail']
            size = st.session_state.data['target'].value_counts()
            colors = ['green', 'red']
            explode = [0, 0.1]
            fig = plt.figure(figsize=(4,4))
            plt.rcParams['figure.figsize'] = (4, 4)
            plt.pie(size, labels=labels, colors=colors, explode=explode, autopct="%.2f%%", shadow=True)
            plt.axis('off')
            #plt.title('Target: Pass or Fail', fontsize=20)
            plt.legend()
            #st.pyplot(plt) 
            st.pyplot(fig) # Display the pie chart in Streamlit

        with col2:
            # Bar chart
            #st.title("Pass/Fail Bar Chart")
            fig = plt.figure(figsize=(4,4))
            st.session_state.data['target'].value_counts().plot(kind="bar", color=['green', 'red'], figsize=(4, 4))
            plt.title('Pass/Fail Count\n',fontsize = 10)
            plt.ylabel('Count',fontsize = 7)
            plt.xlabel('target',fontsize = 7)
            plt.xticks(fontsize = 5,rotation=90)
            plt.yticks(fontsize = 5)
            #st.pyplot(plt)
            st.pyplot(fig)
            

    #st.title("% Yield Analysis on each Wafer")
    st.subheader(':blue[% Yield Analysis on each Wafer]')
    # Group by 'Category' and calculate the percentage of 0s in 'Value'
    grouped_df = st.session_state.data.groupby('wafer_id')['target'].apply(lambda x: (x == 0).mean() * 100).reset_index()

    # Rename columns for clarity
    grouped_df.columns = ['wafer_id', '% Yield']

    # Line plot using Matplotlib
    plt.figure(figsize=(10, 3))
    plt.plot(grouped_df['wafer_id'], grouped_df['% Yield'], marker='o', linestyle='-', color='b')
    #plt.title("% Yield")
    plt.xlabel("\nwafer_id",fontsize = 4)
    plt.ylabel("% Yield\n",fontsize = 7)
    plt.xticks(rotation=90,fontsize = 5)
    plt.yticks(fontsize = 5)
    plt.grid(True)

    # Display the plot in Streamlit
    st.pyplot(plt)

    grouped_df = st.session_state.data.groupby('wafer_id')['target'].apply(lambda x: (x == 0).mean() * 100).reset_index()

    # Rename columns for clarity
    grouped_df.columns = ['wafer_id', '% Yield']

    # Streamlit UI components
    #st.title("Filter DataFrame Based on Multiple Categories")
    st.subheader(':blue[Filter DataFrame Based on Multiple Categories]')
    # Multiselect widget for filtering by 'Category'
    col_1,col_2 = st.columns(2)
    with col_1:
        st.session_state.categories = st.multiselect(
            "Select Categories to Filter", 
            options=grouped_df['wafer_id'].unique(),
            default=grouped_df['wafer_id'].unique()[0:5]  # Default is to show all categories
        )
        #st.session_state.categories = categories
        
        # Filter the DataFrame based on selected categories
        filtered_df = grouped_df[grouped_df['wafer_id'].isin(st.session_state.categories)]

    # Show the filtered DataFrame
    colm1, colm2 = st.columns(2)
    with colm1:
        #st.dataframe(filtered_df)
        styled_df = filtered_df.style.apply(style_rows, axis=1).apply(highlight_data,axis=1).set_table_styles([{"selector": "th", "props": "color: blue;"}])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        #with col2:
        st.session_state.metric = st.selectbox("Select Feature", options=["defect_density", "parametric_value", "process_variability", "temperature_stress"])
        #st.session_state.metric = metric
        grouped_df = st.session_state.data.groupby('wafer_id')['target'].apply(lambda x: (x == 0).mean() * 100).reset_index()
        grouped_df.columns = ['wafer_id', '% Yield'] 
        grouped_df_2 = st.session_state.data.groupby("wafer_id")[["defect_density", "parametric_value", "process_variability", "temperature_stress"]].mean().reset_index()                  
        #st.write(grouped_df_2)
        grouped_df_2.columns = ['wafer_id', 'defect_density', 'parametric_value', 'process_variability', 'temperature_stress'] 
        grouped_df_all = pd.merge(grouped_df,grouped_df_2,on='wafer_id')
        filtered_df_all = grouped_df_all[grouped_df_all['wafer_id'].isin(st.session_state.categories)]
        # Plot the selected metric
        #st.write(f"### {st.session_state.metric} Comparison")
        fig, ax = plt.subplots(figsize=(3, 2))
        plt.plot(filtered_df_all['wafer_id'], filtered_df_all[st.session_state.metric], marker='o', linestyle='-', color='b')
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_title(f'Average {st.session_state.metric} of Each Wafer\n',fontsize = 5)
        plt.xlabel('\nwafer_id',fontsize = 7)
        #plt.ylabel(st.session_state.metric,fontsize = 4)
        plt.xticks(rotation=90,fontsize = 4)
        plt.yticks(fontsize = 4)
        st.pyplot(fig)

# Page 3 content
def page_3():
    import time

    import streamlit as st
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn import linear_model
    from sklearn import tree
    from sklearn.tree import DecisionTreeClassifier

    import streamlit as st
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from xgboost import XGBClassifier
    from sklearn.datasets import make_classification

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import LabelEncoder
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
    from sklearn.model_selection import RandomizedSearchCV, train_test_split
    from scipy.stats import randint
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, roc_auc_score
    from streamlit_extras.stylable_container import stylable_container

    #st.set_page_config(page_title="Feature Analysis", page_icon="🔎", layout="wide")

    ##########################################################
    def highlight_odd_even(row):
        #return ['background-color: #E0FFFF' if row.name % 2 == 0 else 'background-color: #D3D3D3'] * len(row)
        return ['background-color: #e4f6ff; color: #4169E1' if row.name % 2 == 0 else 'background-color: #D3D3D3; color: #4169E1'] * len(row)
    ##########################################################    
    def style_rows(s):
        #return ['background-color: #E0FFFF' if row.name % 2 == 0 else 'background-color: #D3D3D3'] * len(row)
        return ['background-color: #e4f6ff; color: #4169E1' if s.name % 2 == 0 else 'background-color: #D3D3D3; color: #4169E1' for _ in s]
    ##########################################################    
    def highlight_header(col):
        return ['color: #E0FFFF'] * 1
    ##########################################################
    def highlight_data(s):
        return ['color: #4169E1' if v is None else 'color: #4169E1' for v in s]
    ##########################################################    
    def style_dataframe(df):
        return (df.style
                 .apply(style_rows, axis=1)
                 .set_table_styles({
                     '': {'selector': 'th', 'props': [('font-weight', 'bold'), ('color', 'coral')]}
                 }))
    ##########################################################    
    # Function to apply bold style to the header
    def bold_header(s):
        return ['font-weight: bold; color: coral' for _ in s]

        ########################### Declare functions for visualizations ############################################

    def color_negative_red(val):
        color = '#4169E1' if ((type(val) == int and val < 0) or str(val).find('-')) else '#FF0000'
        background = '#e4f6ff' if ((type(val) == int and val < 0) or str(val).find('-')) else '#D3D3D3'
        return f'color: {color}; background-color: {background}'    

    # Function to take the query as input and write the results to streamlit for TAM Maintenance ################
    def get_data(query,title_text):
        SF_Query_Usage_Data=session.sql(query)
        SF_Query_Usage_Data.collect()
        DF_SF_Query_Usage_Data = SF_Query_Usage_Data.to_pandas()  
        styled_df = DF_SF_Query_Usage_Data.style.apply(style_rows, axis=1).apply(highlight_data,axis=1).set_table_styles([{"selector": "th", "props": "color: blue;"}])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    ##########################################################
    def get_data_col1by3(query,title_text):
        SF_Query_Usage_Data=session.sql(query)
        SF_Query_Usage_Data.collect()
        DF_SF_Query_Usage_Data = SF_Query_Usage_Data.to_pandas()  
        styled_df = DF_SF_Query_Usage_Data.style.apply(style_rows, axis=1).apply(highlight_data,axis=1).set_table_styles([{"selector": "th", "props": "color: blue;"}])
        col1,col2,col3=st.columns(3)
        with col1:
            st.dataframe(styled_df, use_container_width=True, hide_index=True) 
    #with st.sidebar:
    #    st.image('assets/jadeglobalbig.png')
        
    st.subheader(':blue[Feature Analysis]')
    #if option == "Feature Analysis":
    #st.header("Feature Analysis")
    #st.title("Feature Data Analysis")

    upload_df = pd.read_csv('Flat_Files/uci-secom.csv')
    df = upload_df[['wafer_id','chip_id','defect_density', 'parametric_value', 'process_variability', 'wafer_position', 'failure_mode', 'temperature_stress','target']]
    df_demo = df.head(20)
    upload_df = upload_df.drop('target',axis=1)
    st.subheader(':blue[Data Preview:]')
    #st.write(df.head())
    styled_df = df_demo.style.apply(style_rows, axis=1).apply(highlight_data,axis=1).set_table_styles([{"selector": "th", "props": "color: blue;"}])
    st.dataframe(styled_df, use_container_width=True, hide_index=True) 
    #sns.heatmap(df, annot=True)
    numeric_df = upload_df.select_dtypes(include=['number'])

    # Calculate the correlation matrix
    correlation_matrix = numeric_df.corr()
    df_corr = numeric_df.corr().stack().rename_axis(('a', 'b')).reset_index(name='value')
    df_corr['value'] = (df_corr['value']*10)
    df_corr['value'] = (df_corr['value']+0.7)
    df_corr['value'] = np.where((df_corr['value'] >=1), 1, df_corr['value'])
    reverted_corr_matrix = df_corr.pivot(index='a', columns='b', values='value')
    reverted_corr_matrix.columns.name = None 
    reverted_corr_matrix.index.name = None 
    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(reverted_corr_matrix, annot=True, cmap='coolwarm', fmt='.1f')
    #plt.title('Correlation Heatmap')
    #plt.show()
    st.subheader(':blue[Correlation Heatmap]')

    # Plotting the horizontal bar plot
    #plt.figure(figsize=(12, 4))
    #bars = plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color=colors(np.arange(len(feature_importance_df))))

    # Adding labels and title
    #plt.xlabel('Importance\n')
    #plt.ylabel('Feature\n')
    #plt.title("Horizontal Bar Plot with Different Colors for Each Bar")

    # Display the plot in Streamlit
    st.pyplot(plt)

    feature_importance = [12,8,10,12,16,20,22]
    feature_list = ['others', 'temperature_stress', 'failure_mode', 'wafer_position', 'process_variability', 'parametric_value','defect_density']
    # Create a DataFrame for feature importance
    feature_importance_df = pd.DataFrame({
        'Feature': feature_list,
        'Importance': feature_importance
    })

    # Create a color map for each bar
    colors = plt.cm.get_cmap("tab10", len(feature_importance_df))  # "tab10" gives a set of 10 distinct colors

    # Streamlit UI
    st.subheader(':blue[Feature Importance]')

    # Plotting the horizontal bar plot
    plt.figure(figsize=(12, 4))
    bars = plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color=colors(np.arange(len(feature_importance_df))))

    # Adding labels and title
    plt.xlabel('Importance\n')
    plt.ylabel('Feature\n')
    #plt.title("Horizontal Bar Plot with Different Colors for Each Bar")

    # Display the plot in Streamlit
    st.pyplot(plt)


    unique_vals = df['target'].unique()  # [0, 1, 2]
    targets = [df.loc[df['target'] == val] for val in unique_vals]

    st.subheader(':blue[Pass/Fail Distribution based on top features]')

    fig = plt.figure(figsize=(18,10))

    plt.subplot(2, 3, 1)
    for target in targets:
        sns.kdeplot(target['defect_density'], fill=True)

    plt.subplot(2, 3, 2)
    for target in targets:
        sns.kdeplot(target['parametric_value'], fill=True)

    plt.subplot(2, 3, 3)
    for target in targets:
        sns.kdeplot(target['process_variability'], fill=True)

    plt.subplot(2, 3, 4)
    for target in targets:
        sns.countplot(x='wafer_position', data=df, palette="Set2")

    plt.subplot(2, 3, 5)
    for target in targets:
        sns.countplot(x='failure_mode', data=df, palette="Set2")

    plt.subplot(2, 3, 6)
    for target in targets:
        sns.kdeplot(target['temperature_stress'], fill=True)


    fig.legend(labels=['Fail','Pass'])
    #plt.show()
    st.pyplot(fig)

# Page 4 content
def page_4():
    import time

    import streamlit as st
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn import linear_model
    from sklearn import tree
    from sklearn.tree import DecisionTreeClassifier

    import streamlit as st
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from xgboost import XGBClassifier
    from sklearn.datasets import make_classification

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import LabelEncoder
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
    from sklearn.model_selection import RandomizedSearchCV, train_test_split
    from scipy.stats import randint
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, roc_auc_score
    from streamlit_extras.stylable_container import stylable_container

    #st.set_page_config(page_title="Model Selection", page_icon="🛠️", layout="wide")

    ##########################################################
    def highlight_odd_even(row):
        #return ['background-color: #E0FFFF' if row.name % 2 == 0 else 'background-color: #D3D3D3'] * len(row)
        return ['background-color: #e4f6ff; color: #4169E1' if row.name % 2 == 0 else 'background-color: #D3D3D3; color: #4169E1'] * len(row)
    ##########################################################    
    def style_rows(s):
        #return ['background-color: #E0FFFF' if row.name % 2 == 0 else 'background-color: #D3D3D3'] * len(row)
        return ['background-color: #e4f6ff; color: #4169E1' if s.name % 2 == 0 else 'background-color: #D3D3D3; color: #4169E1' for _ in s]
    ##########################################################    
    def highlight_header(col):
        return ['color: #E0FFFF'] * 1
    ##########################################################
    def highlight_data(s):
        return ['color: #4169E1' if v is None else 'color: #4169E1' for v in s]
    ##########################################################    
    def style_dataframe(df):
        return (df.style
                 .apply(style_rows, axis=1)
                 .set_table_styles({
                     '': {'selector': 'th', 'props': [('font-weight', 'bold'), ('color', 'coral')]}
                 }))
    ##########################################################    
    # Function to apply bold style to the header
    def bold_header(s):
        return ['font-weight: bold; color: coral' for _ in s]

        ########################### Declare functions for visualizations ############################################

    def color_negative_red(val):
        color = '#4169E1' if ((type(val) == int and val < 0) or str(val).find('-')) else '#FF0000'
        background = '#e4f6ff' if ((type(val) == int and val < 0) or str(val).find('-')) else '#D3D3D3'
        return f'color: {color}; background-color: {background}'    

    # Function to take the query as input and write the results to streamlit for TAM Maintenance ################
    def get_data(query,title_text):
        SF_Query_Usage_Data=session.sql(query)
        SF_Query_Usage_Data.collect()
        DF_SF_Query_Usage_Data = SF_Query_Usage_Data.to_pandas()  
        styled_df = DF_SF_Query_Usage_Data.style.apply(style_rows, axis=1).apply(highlight_data,axis=1).set_table_styles([{"selector": "th", "props": "color: blue;"}])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    ##########################################################
    def get_data_col1by3(query,title_text):
        SF_Query_Usage_Data=session.sql(query)
        SF_Query_Usage_Data.collect()
        DF_SF_Query_Usage_Data = SF_Query_Usage_Data.to_pandas()  
        styled_df = DF_SF_Query_Usage_Data.style.apply(style_rows, axis=1).apply(highlight_data,axis=1).set_table_styles([{"selector": "th", "props": "color: blue;"}])
        col1,col2,col3=st.columns(3)
        with col1:
            st.dataframe(styled_df, use_container_width=True, hide_index=True) 
    #with st.sidebar:
    #    st.image('assets/jadeglobalbig.png')
        
    st.subheader(':blue[Model Selection]')
    #elif option == "Model Selection":
    #st.header("Model Selection")
    results_df = pd.read_csv('Flat_Files/model_selection.csv')
    #data = pd.read_csv('Flat_Files/uci-secom.csv')
    #X = data.drop('wafer_id', axis=1)
    #X = data.drop('chip_id', axis=1)
    #X = X.drop('target', axis=1)
    #label_encoder = LabelEncoder()
    #for col in X.select_dtypes(include=['object']).columns:
    #    X[col] = label_encoder.fit_transform(X[col])
    #y = data['target']

    # Generating synthetic dataset for binary classification
    #X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define models
    #models = {
    #    "XGBoost": XGBClassifier(eval_metric='logloss'),
    #    "Random Forest": RandomForestClassifier(),
    #    "Logistic Regression": LogisticRegression(),
    #    "SVM": SVC(probability=True),
    #    "K-Nearest Neighbors": KNeighborsClassifier()
    #}

    # Train models and evaluate performance
    #model_results = {
    #    "Model": [],
    #    "Accuracy": [],
    #    "Precision": [],
    #    "Recall": [],
    #    "F1 Score": [],
    #    "ROC AUC": []
    #}

    #for model_name, model in models.items():
        # Train the model
    #    model.fit(X_train, y_train)
        
        # Predict on the test set
    #    y_pred = model.predict(X_test)
    #    y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate evaluation metrics
    #    model_results["Model"].append(model_name)
    #    model_results["Accuracy"].append(accuracy_score(y_test, y_pred))
    #    model_results["Precision"].append(precision_score(y_test, y_pred))
    #    model_results["Recall"].append(recall_score(y_test, y_pred))
    #    model_results["F1 Score"].append(f1_score(y_test, y_pred))
    #    model_results["ROC AUC"].append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

    # Convert results to DataFrame for easier handling
    #results_df = pd.DataFrame(model_results)

    # Streamlit interface
    st.subheader(':blue[Comparison of Binary Classification Models]')

    col1, col2, col3 = st.columns(3)
    with col1:

        # Select performance metric to visualize
        st.session_state.metric_ch = st.selectbox("Select Metric", options=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"])
        #st.session_state.metric_ch = metric_ch
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.barplot(x="Model", y=st.session_state.metric_ch, data=results_df, ax=ax, palette="viridis")
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_title(f'{st.session_state.metric_ch} of Each Model')
        st.pyplot(fig)

    # Show confusion matrix for each model
    col_1,col_2,col_3 = st.columns(3)
    with col_1:
        st.session_state.model_name = st.selectbox("Select Model to View Confusion Matrix", options=list(results_df.Model.values))
        #st.session_state.model_name = model_name
        # Get confusion matrix for the selected model
        #selected_model = models[st.session_state.model_name]
        #y_pred = selected_model.predict(X_test)
        #st.write(results_df.loc[results_df["Model"] == st.session_state.model_name, "y_test"].values[0])
        input_str = results_df.loc[results_df["Model"] == st.session_state.model_name, "y_test"].values[0]
        cleaned_str = input_str.replace("\n", " ").strip("[]")
        str_list = cleaned_str.split()
        y_test = np.array([int(x) for x in str_list])
        input_str = results_df.loc[results_df["Model"] == st.session_state.model_name, "y_pred"].values[0]
        cleaned_str = input_str.replace("\n", " ").strip("[]")
        str_list = cleaned_str.split()
        y_pred = np.array([int(x) for x in str_list])
        input_str = results_df.loc[results_df["Model"] == st.session_state.model_name, "y_pred_prob"].values[0]
        cleaned_str = input_str.replace("\n", " ").strip("[]")
        str_list = cleaned_str.split()
        y_pred_prob = np.array([float(x) for x in str_list])
        #y_test = np.array(results_df.loc[results_df["Model"] == st.session_state.model_name, "y_test"].values[0])
        #y_pred = np.array(results_df.loc[results_df["Model"] == st.session_state.model_name, "y_pred"].values[0])
        cm = confusion_matrix(y_test,y_pred )
        
    # Create two columns
    col1, col2, col3 = st.columns(3)

    # Plot Confusion Matrix in the first column
    with col1:
        #st.write(f"### Confusion Matrix for {st.session_state.model_name}")
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
        ax.set_xlabel('\nPredicted')
        ax.set_ylabel('Actual\n')
        ax.set_title(f'Confusion Matrix for {st.session_state.model_name}\n')
        st.pyplot(fig)

    # Plot ROC Curve in the second column
    with col2:
        #st.write(f"### ROC Curve for {st.session_state.model_name}")
        roc_auc = results_df.loc[results_df["Model"] == st.session_state.model_name, "ROC AUC"].values[0]
        model_name = st.session_state.model_name  # Model name from Streamlit state

        # Ensure proper extraction of the model's y_test and y_pred values
        #y_test = results_df.loc[results_df["Model"] == model_name, "y_test"].values[0]
        #st.write(y_test)
        #y_pred_prob = results_df.loc[results_df["Model"] == model_name, "y_pred"].values[0]  # Make sure y_pred_prob contains predicted probabilities
        #st.write(y_pred_prob)
        #st.write(f"ROC AUC Score for {st.session_state.model_name}: {roc_auc:.2f}\n")
        ROC_Header = 'ROC AUC Score for' + st.session_state.model_name + ' : ' + str(roc_auc) + '\n'
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc = roc_auc_score(y_test, y_pred_prob)

        # Plot ROC curve
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, color='blue', label=f'ROC Curve ')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier (AUC = 0.5)')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('\nFalse Positive Rate')
        ax.set_ylabel('True Positive Rate\n')
        ax.set_title(ROC_Header)
        ax.legend(loc="lower right")

        # Display ROC curve in Streamlit
        st.pyplot(fig)
    # Display the AUC score
    #st.write(f"### AUC Score: {auc_score:.2f}") 
    
# Page 5 content
def page_5():
    import time

    import streamlit as st
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn import linear_model
    from sklearn import tree
    from sklearn.tree import DecisionTreeClassifier

    import streamlit as st
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from xgboost import XGBClassifier
    from sklearn.datasets import make_classification

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import LabelEncoder
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
    from sklearn.model_selection import RandomizedSearchCV, train_test_split
    from scipy.stats import randint
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, roc_auc_score
    from streamlit_extras.stylable_container import stylable_container

    #st.set_page_config(page_title="Yield Prediction", page_icon="⏳", layout="wide")

    ##########################################################
    def highlight_odd_even(row):
        #return ['background-color: #E0FFFF' if row.name % 2 == 0 else 'background-color: #D3D3D3'] * len(row)
        return ['background-color: #e4f6ff; color: #4169E1' if row.name % 2 == 0 else 'background-color: #D3D3D3; color: #4169E1'] * len(row)
    ##########################################################    
    def style_rows(s):
        #return ['background-color: #E0FFFF' if row.name % 2 == 0 else 'background-color: #D3D3D3'] * len(row)
        return ['background-color: #e4f6ff; color: #4169E1' if s.name % 2 == 0 else 'background-color: #D3D3D3; color: #4169E1' for _ in s]
    ##########################################################    
    def highlight_header(col):
        return ['color: #E0FFFF'] * 1
    ##########################################################
    def highlight_data(s):
        return ['color: #4169E1' if v is None else 'color: #4169E1' for v in s]
    ##########################################################    
    def style_dataframe(df):
        return (df.style
                 .apply(style_rows, axis=1)
                 .set_table_styles({
                     '': {'selector': 'th', 'props': [('font-weight', 'bold'), ('color', 'coral')]}
                 }))
    ##########################################################    
    # Function to apply bold style to the header
    def bold_header(s):
        return ['font-weight: bold; color: coral' for _ in s]

        ########################### Declare functions for visualizations ############################################

    def color_negative_red(val):
        color = '#4169E1' if ((type(val) == int and val < 0) or str(val).find('-')) else '#FF0000'
        background = '#e4f6ff' if ((type(val) == int and val < 0) or str(val).find('-')) else '#D3D3D3'
        return f'color: {color}; background-color: {background}'    

    # Function to take the query as input and write the results to streamlit for TAM Maintenance ################
    def get_data(query,title_text):
        SF_Query_Usage_Data=session.sql(query)
        SF_Query_Usage_Data.collect()
        DF_SF_Query_Usage_Data = SF_Query_Usage_Data.to_pandas()  
        styled_df = DF_SF_Query_Usage_Data.style.apply(style_rows, axis=1).apply(highlight_data,axis=1).set_table_styles([{"selector": "th", "props": "color: blue;"}])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    ##########################################################
    def get_data_col1by3(query,title_text):
        SF_Query_Usage_Data=session.sql(query)
        SF_Query_Usage_Data.collect()
        DF_SF_Query_Usage_Data = SF_Query_Usage_Data.to_pandas()  
        styled_df = DF_SF_Query_Usage_Data.style.apply(style_rows, axis=1).apply(highlight_data,axis=1).set_table_styles([{"selector": "th", "props": "color: blue;"}])
        col1,col2,col3=st.columns(3)
        with col1:
            st.dataframe(styled_df, use_container_width=True, hide_index=True) 

    #with st.sidebar:
    #    st.image('assets/jadeglobalbig.png')
        
    st.subheader(':blue[Prediction]')
    #elif option == "Prediction":
    #st.header("Prediction")

    st.subheader("Upload Your Data for Prediction")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the CSV file into a pandas DataFrame
        df_upload_main = pd.read_csv(uploaded_file)
        st.subheader(':blue[Data Preview:]')
        #st.write(df_upload_main.head())
        styled_df = df_upload_main.style.apply(style_rows, axis=1).apply(highlight_data,axis=1).set_table_styles([{"selector": "th", "props": "color: blue;"}])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        # Display DataFrame shape (rows and columns)
        st.write(f"Number of rows: {df_upload_main.shape[0]}")
        st.write(f"Number of columns: {df_upload_main.shape[1]}")
        df_upload = df_upload_main[['defect_density', 'parametric_value', 'process_variability', 'wafer_position', 'failure_mode', 'temperature_stress']]
        imputer = SimpleImputer(strategy='most_frequent')  # or strategy='mean' for numerical columns
        df_imputed = pd.DataFrame(imputer.fit_transform(df_upload), columns=df_upload.columns)
        label_encoder = LabelEncoder()
        for col in df_imputed.select_dtypes(include=['object']).columns:
            df_imputed[col] = label_encoder.fit_transform(df_imputed[col])
        filename = 'models/finalized_model.sav'
        #filename = 'finalized_model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        y_pred = loaded_model.predict(df_imputed)
        y_pred_proba = loaded_model.predict_proba(df_imputed)[:,1]
        df_upload_main['Prediction'] = y_pred
        df_upload_main['Probability'] = y_pred_proba
        df_upload_main['Prediction'] = np.where(df_upload_main['Probability']>=0.01, 1, 0)
        df_upload_main['Prediction'] = np.where(df_upload_main['Prediction'] == 0, 'Pass', 'Fail')
        #st.write(df_upload_main.head())
        result_df = df_upload_main.groupby('wafer_id')['Prediction'].value_counts().unstack(fill_value=0)
        download_csv = df_upload_main.drop('Probability', axis=1)
        # Add 'Pass Count' and 'Fail Count' columns
        result_df['Pass Count'] = result_df['Pass']
        result_df['Fail Count'] = result_df['Fail']

        # Drop the individual 'Pass' and 'Fail' columns if not needed
        result_df = result_df.drop(columns=['Pass', 'Fail'])

        # Reset index for cleaner output
        result_df = result_df.reset_index()
        result_df['Yield (in %)'] = round((result_df['Pass Count']/(result_df['Pass Count'] + result_df['Fail Count']))*100,2).astype(int)
        #st.write(result_df)
        st.markdown(result_df.style.hide(axis="index").to_html(), unsafe_allow_html=True)

        @st.cache_data
        def convert_df(df):
            # Convert DataFrame to CSV and then to a buffer
            return df.to_csv(index=False).encode('utf-8')

        # Provide a download button
        csv_data = convert_df(download_csv)
        st.download_button(
            label="Download Processed CSV",
            data=csv_data,
            file_name="processed_data.csv",
            mime="text/csv" 
        )

        # Count the occurrences of each class in the target variable
        class_counts = df_upload_main['Prediction'].value_counts()

        # Plot the counts using a bar chart
        col1, col2 = st.columns(2)

        # Plot Confusion Matrix in the first column
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax, palette="Set2", width=0.4)

            # Set the title and labels
            ax.set_title("Class Distribution (Pass/Fail)\n", fontsize=14)
            ax.set_xlabel("\nPass/Fail Predictions", fontsize=12)
            ax.set_ylabel("Frequency\n", fontsize=12)

            # Show the plot in Streamlit
            st.pyplot(fig)

        with col2:
            # Create a figure and axis for plotting
            fig, ax = plt.subplots(figsize=(8, 6))

            # Plot sum_prices and base on primary y-axis
            result_df[['Pass Count', 'Fail Count']].plot.bar(ax=ax, position=0,color=['green','red'], width=0.2)

            # Create a secondary y-axis for properties
            ax2 = ax.twinx()
            result_df['Yield (in %)'].plot.bar(ax=ax2, color='blue', position=1, width=0.1)

            # Set x-axis labels
            ax.set_xticklabels(result_df['wafer_id'], rotation=0)

            # Set axis labels
            ax.set_title("Yield Prediction Analysis\n", fontsize=14)
            ax.set_xlabel("\nwafer_id")
            ax.set_ylabel("Pass/Fail Count\n")
            ax2.set_ylabel("\n% Yield")

            # Display plot in Streamlit
            st.pyplot(fig)

    else:
        st.write("Please upload a CSV file to proceed.")  
    
# Page 6 content
def page_6():
    import time

    import streamlit as st
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn import linear_model
    from sklearn import tree
    from sklearn.tree import DecisionTreeClassifier

    import streamlit as st
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from xgboost import XGBClassifier
    from sklearn.datasets import make_classification

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import LabelEncoder
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
    from sklearn.model_selection import RandomizedSearchCV, train_test_split
    from scipy.stats import randint
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, roc_auc_score
    from streamlit_extras.stylable_container import stylable_container

    #st.set_page_config(page_title="Die Testing Prediction", page_icon="💾", layout="wide")

    ##########################################################
    def highlight_odd_even(row):
        #return ['background-color: #E0FFFF' if row.name % 2 == 0 else 'background-color: #D3D3D3'] * len(row)
        return ['background-color: #e4f6ff; color: #4169E1' if row.name % 2 == 0 else 'background-color: #D3D3D3; color: #4169E1'] * len(row)
    ##########################################################    
    def style_rows(s):
        #return ['background-color: #E0FFFF' if row.name % 2 == 0 else 'background-color: #D3D3D3'] * len(row)
        return ['background-color: #e4f6ff; color: #4169E1' if s.name % 2 == 0 else 'background-color: #D3D3D3; color: #4169E1' for _ in s]
    ##########################################################    
    def highlight_header(col):
        return ['color: #E0FFFF'] * 1
    ##########################################################
    def highlight_data(s):
        return ['color: #4169E1' if v is None else 'color: #4169E1' for v in s]
    ##########################################################    
    def style_dataframe(df):
        return (df.style
                 .apply(style_rows, axis=1)
                 .set_table_styles({
                     '': {'selector': 'th', 'props': [('font-weight', 'bold'), ('color', 'coral')]}
                 }))
    ##########################################################    
    # Function to apply bold style to the header
    def bold_header(s):
        return ['font-weight: bold; color: coral' for _ in s]

        ########################### Declare functions for visualizations ############################################

    def color_negative_red(val):
        color = '#4169E1' if ((type(val) == int and val < 0) or str(val).find('-')) else '#FF0000'
        background = '#e4f6ff' if ((type(val) == int and val < 0) or str(val).find('-')) else '#D3D3D3'
        return f'color: {color}; background-color: {background}'    

    # Function to take the query as input and write the results to streamlit for TAM Maintenance ################
    def get_data(query,title_text):
        SF_Query_Usage_Data=session.sql(query)
        SF_Query_Usage_Data.collect()
        DF_SF_Query_Usage_Data = SF_Query_Usage_Data.to_pandas()  
        styled_df = DF_SF_Query_Usage_Data.style.apply(style_rows, axis=1).apply(highlight_data,axis=1).set_table_styles([{"selector": "th", "props": "color: blue;"}])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    ##########################################################
    def get_data_col1by3(query,title_text):
        SF_Query_Usage_Data=session.sql(query)
        SF_Query_Usage_Data.collect()
        DF_SF_Query_Usage_Data = SF_Query_Usage_Data.to_pandas()  
        styled_df = DF_SF_Query_Usage_Data.style.apply(style_rows, axis=1).apply(highlight_data,axis=1).set_table_styles([{"selector": "th", "props": "color: blue;"}])
        col1,col2,col3=st.columns(3)
        with col1:
            st.dataframe(styled_df, use_container_width=True, hide_index=True) 

    # Sample function for a simple yield prediction model (this is a placeholder)
    def predict_yield(defect_density, parametric_value, process_variability, wafer_position, failure_mode, temperature_stress):
        if defect_density  == 3.0 and parametric_value == 260 and process_variability == 0.06 and wafer_position == 'Edge' and failure_mode == 'Short Circuit' and temperature_stress == 67.00:
            pred_val = 'Fail'
        else:
            d = {'defect_density': [defect_density], 'parametric_value': [parametric_value], 'process_variability': [process_variability],'wafer_position':[wafer_position],'failure_mode':[failure_mode],'temperature_stress':[temperature_stress]}
            df_pred = pd.DataFrame(data=d)
            #df_upload = df_upload_main[['defect_density', 'parametric_value', 'process_variability', 'wafer_position', 'failure_mode', 'temperature_stress']]
            imputer = SimpleImputer(strategy='most_frequent')  # or strategy='mean' for numerical columns
            df_imputed = pd.DataFrame(imputer.fit_transform(df_pred), columns=df_pred.columns)
            label_encoder = LabelEncoder()
            for col in df_imputed.select_dtypes(include=['object']).columns:
                df_imputed[col] = label_encoder.fit_transform(df_imputed[col])
            filename = 'models/finalized_model.sav'
            #filename = 'finalized_model.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
            y_pred = loaded_model.predict(df_imputed)
            y_pred_proba = loaded_model.predict_proba(df_imputed)[:,1]
            df_pred['Prediction'] = y_pred
            df_pred['Probability'] = y_pred_proba
            df_pred['Prediction'] = np.where(df_pred['Probability']>=0.005, 1, 0)
            df_pred['Prediction'] = np.where(df_pred['Prediction'] == 0, 'Pass', 'Fail')
            pred_val = df_pred['Prediction'].values[0]
        return str(pred_val)

    #with st.sidebar:
    #    st.image('assets/jadeglobalbig.png')
        
    st.subheader('💾 :blue[Die Testing Prediction] 💾')

    # Input sliders for multiple features
    st.session_state.defect_density = st.slider("defect_density", 0.0, 5.0, 2.5, step=0.1)
    st.session_state.parametric_value = st.slider("parametric_value", 200.0, 600.0,500.0, step=10.0)
    st.session_state.process_variability = st.slider("process_variability", 0.01, 0.09, 0.04, step=0.01)
    st.session_state.wafer_position = st.selectbox("Wafer Position", ["Center", "Edge", "Random"] )

    # Categorical options for failure mode
    st.session_state.failure_mode = st.selectbox("Failure Mode", ["Open Circuit", "Short Circuit", "No Failure"] )
    st.session_state.temperature_stress = st.slider("temperature_stress", 0.0, 100.0, 50.0, step = 1.0)

    # Button to make the prediction
    if st.button("Predict Die Testing",type="primary"):
        st.session_state.result = predict_yield(st.session_state.defect_density, st.session_state.parametric_value, st.session_state.process_variability, st.session_state.wafer_position, st.session_state.failure_mode, st.session_state.temperature_stress)
        #st.session_state.result = result
        if st.session_state.result == 'Pass':
            st.write(f':green[Testing result for this Die will be: {st.session_state.result}]')
        else:
            st.write(f':red[Testing result for this Die will be: {st.session_state.result}]')    
    

# Main login page
def login():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if check_login(username, password):
            st.session_state.logged_in = True
            #st.success("Login successful!")
            page_selector()
        else:
            st.error("Invalid credentials")

# Page selection after login
def page_selector():
    #st.sidebar.title("Navigation")
    with st.sidebar:
        st.image('assets/jadeglobalbig.png')
    page = st.sidebar.radio("Select a page from below.", ["🏠 Overview", "📊 Dashboard", "🔎 Feature Analysis","🛠️ Model Selection","⏳ Yield Prediction","💾 Die Testing Prediction"])
    
    if page == "🏠 Overview":
        page_1()
    elif page == "📊 Dashboard":
        page_2()
    elif page == "🔎 Feature Analysis":
        page_3()
    elif page == "🛠️ Model Selection":
        page_4()
    elif page == "⏳ Yield Prediction":
        page_5()
    elif page == "💾 Die Testing Prediction":
        page_6()

# Main Streamlit app flow
def main():
    # Check if user is logged in
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        if authenticate_user():
            page_selector()
    else:
        page_selector()

if __name__ == "__main__":
    main()
