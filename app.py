import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import pickle
import plotly.express as px

st.set_page_config(
    page_title="Loan Approval App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global variables
df_predicted = pd.DataFrame()
uploaded_file = None

# Session state initialization
if 'df_input' not in st.session_state or uploaded_file is None:
    st.session_state['df_input'] = pd.DataFrame()

if 'df_predicted' not in st.session_state:
    st.session_state['df_predicted'] = pd.DataFrame()

if 'tab_selected' not in st.session_state:
    st.session_state['tab_selected'] = None
    
def reset_session_state():
    st.session_state['df_input'] = pd.DataFrame()
    st.session_state['df_predicted'] = pd.DataFrame()
    
# ML Section
numerical = ['no_of_dependents', 'income_annum', 'loan_amount', 
             'loan_term', 'cibil_score', 'residential_assets_value', 
             'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']
categorical = ['education', 'self_employed']
le_enc_cols = ['education', 'self_employed']
education_map = {'not_graduate': 0, 'graduate': 1}
y_n_map = {'yes': 1, 'no': 0}

# Logistic regression model
model_file_path = 'C:/Users/Admin/Documents/AI/поток 9/AI/lp/models/lr_model_loan_status.sav'
model = pickle.load(open(model_file_path, 'rb'))

# Encoding model DictVectorizer
encoding_model_file_path = 'C:/Users/Admin/Documents/AI/поток 9/AI/lp/models/encoding_loan_status.sav'
encoding_model = pickle.load(open(encoding_model_file_path, 'rb'))

@st.cache_data
def predict_loan(df_input, treshold):

    scaler = MinMaxScaler()

    df_original = df_input.copy()
    df_input[numerical] = scaler.fit_transform(df_input[numerical])

    for col in le_enc_cols:
        if col == 'education':
            df_input[col] = df_input[col].map(education_map)
        else:
            df_input[col] = df_input[col].map(y_n_map)

    dicts_df = df_input[categorical + numerical].to_dict(orient='records')
    X = encoding_model.transform(dicts_df)
    X[np.isnan(X)] = 0
    y_pred = model.predict_proba(X)[:, 1]
    loan_descision = (y_pred >= treshold).astype(int)
    df_original['loan_predicted'] = loan_descision
    df_original['loan_predicted_probability'] = y_pred

    return df_original

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Sidebar section
with st.sidebar:
    st.title('🗂 Ввод данных')
    tab1, tab2 = st.tabs(['📁 Данные из файла', '📝 Ввести вручную'])
    with tab1:
        uploaded_file = st.file_uploader("Выбрать CSV файл", type=['csv', 'xlsx'], on_change=reset_session_state)
        if uploaded_file is not None:
            treshold = st.slider("Порог вероятности одобрения кредита", 0.0, 1.0, 0.5, 0.01, key="slider1")
            prediction_button = st.button("Предсказать", type="primary", use_container_width=True, key="button1")
            st.session_state['df_input'] = pd.read_csv(uploaded_file)
            if prediction_button:
                st.session_state['df_predicted'] = predict_loan(st.session_state['df_input'], treshold)
                st.session_state['tab_selected'] = "tab1"
                
    with tab2:
        loan_id = st.text_input('Loan ID', placeholder='0000', help='Введите Id заявки')
        no_of_depends = st.number_input('Количество иждивенцев', min_value=0, value=0)
        education = st.selectbox('Образование', ('Graduate', 'Not Graduate'))
        self_employed = st.selectbox('Статус занятности', ('Yes', 'No'))
        income_annum = st.number_input('Годовой доход', min_value=0, value=0)
        loan_amount = st.number_input('Сумма кредита', min_value=1, value=1)
        loan_term = st.number_input('Срок кредита', min_value=1, value=1)
        cibil_score = st.number_input('Кредитный рейтинг', min_value=0, value=600)
        residental_assets_value = st.number_input('Cтоимость жилого фонда', min_value=0, value=0)
        commercial_assets_value = st.number_input('Стоимость коммерческих активов', min_value=0, value=0)
        luxury_assets_value = st.number_input('Стоимость люксовых активов', min_value=0, value=0)
        bank_assets_value = st.number_input('Стоимость активов банка', min_value=0, value=0)

        if loan_id != "":
            treshold = st.slider("Статус вероятности отказа", 0.0, 1.0, 0.5, 0.01, key='slider2')
            prediction_button_tab2 = st.button('Предсказать', type="primary", use_container_width=True, key="button2")
            
            if prediction_button_tab2:
                st.session_state['tab_selected'] = 'tab2'               
                st.session_state['df_input'] = pd.DataFrame({
                    'loan_id': loan_id,
                    'education': 1 if education == 'Graduate' else 0,
                    'self_employed': 1 if self_employed == 'Yes' else 0,
                    'no_of_dependents': int(no_of_depends),
                    'income_annum': int(income_annum),
                    'loan_amount': int(loan_amount),
                    'loan_term': int(loan_term),
                    'cibil_score': int(cibil_score),
                    'residential_assets_value': int(residental_assets_value),
                    'commercial_assets_value': int(commercial_assets_value),
                    'luxury_assets_value': int(luxury_assets_value),
                    'bank_asset_value': int(bank_assets_value)                    
                }, index=[0])
                st.session_state['df_predicted'] = predict_loan(st.session_state['df_input'], treshold)
                
# Sidebar section end
    
# Main section start
st.image('https://thumbor.forbes.com/thumbor/fit-in/900x510/https://www.forbes.com/advisor/ca/wp-content/uploads/2023/09/how-to-get-guaranteed-approval-for-a-personal-loan-ca-e1695193850403.jpg', width=600)
st.title('Прогнозирование одобрение кредита')

with st.expander("Описание проекта"):
    st.write("""
    В данном проекте мы рассмотрим задачу по одобрению кредита.
    Для этого мы будем использовать датасет из открытых источников.
    Датасет содержит информацию о заявках, которым одобрили или отклонили кредит.
    Наша задача - построить модель, которая будет предсказывать одобрение заявки.
    """)

if len(st.session_state['df_input']) > 0:
    if len(st.session_state['df_predicted']) == 0:
        st.subheader("Данные из файла")
        st.write(st.session_state['df_input'])
    else:
        with st.expander("Входные данные"):
            st.write(st.session_state['df_input'])

if len(st.session_state['df_predicted']) > 0 and st.session_state['tab_selected'] == 'tab2':
    if st.session_state['df_predicted']['loan_predicted'][0] == 0:
        st.subheader(f"Заявку :red[отказали] с вероятностью {(1 - st.session_state["df_predicted"]["loan_predicted_probability"][0]) * 100:.2f}%")
    else:
        st.subheader(f"Клиенту :green[одобрили] {st.session_state["df_predicted"]["loan_predicted_probability"][0] * 100:.2f}%")
    

    
if len(st.session_state['df_predicted']) > 1:
    st.subheader("Результаты прогнозирования")
    st.write(st.session_state['df_predicted'])
    res_all_csv = convert_df(st.session_state['df_predicted'])
    st.download_button(
        label="Скачать все предсказания",
        data=res_all_csv,
        file_name='df-loan-predicted-all.csv',
        mime='text/csv',
    )
    
    fig = px.histogram(st.session_state['df_predicted'], x='loan_predicted', color = 'loan_predicted')
    st.plotly_chart(fig, use_container_width=True)

    risk_clients = st.session_state['df_predicted'][st.session_state['df_predicted']['loan_predicted'] == 1]

    if len(risk_clients) > 0:
        st.subheader("Заявки с высоким риском отказа")
        st.write(risk_clients)
    
        res_risky_csv = convert_df(risk_clients)
        st.download_button(
            label="Скачать заявки с высоким риском отказа",
            data=res_all_csv,
            file_name='df-loan-predicted-risk.csv',
            mime='text/csv',
        )
