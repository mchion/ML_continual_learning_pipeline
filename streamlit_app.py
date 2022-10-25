import streamlit as st
import pandas as pd
from google.oauth2 import service_account
from google.cloud import bigquery

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from PIL import Image

import torch
from torch.autograd import Variable
import torch.nn.functional as nnf
from torchvision.transforms import Compose, ToTensor, Resize

st.set_page_config(
    page_title='Archdaily.com Image Classifier',
    layout='wide'
)


# Create connection to BigQuery database
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)

def dataset_classes():
    
    # Get initial dataset class distribution
    query_initial_model = (
            """
            SELECT T.continent, sum(T.image_count) AS i_count
            FROM
                (
                SELECT continent, date_added,
                        (
                            COALESCE((CASE WHEN image_1 IS NOT NULL THEN 1 ELSE 0 END), 0)
                            + COALESCE((CASE WHEN image_2 IS NOT NULL THEN 1 ELSE 0 END), 0)
                            + COALESCE((CASE WHEN image_3 IS NOT NULL THEN 1 ELSE 0 END), 0)
                            ) AS image_count
                        FROM `archdaily_dataset.table-post-id`
                        ) as T
                        WHERE date_added = timestamp('2022-10-18')
            GROUP BY T.continent
            HAVING continent IS NOT NULL
            """
        )
    df = client.query(query_initial_model).to_dataframe()
    df['percentage'] = df['i_count'].astype(str) + " (" + (df['i_count'] / df['i_count'].sum()).apply(lambda x: '{:,.1%}'.format(x)).astype(str) + ")"
    total_images1 = df['i_count'].sum()

    # Get current dataset class distribution
    query_current_model = (
            """
            SELECT T.continent, sum(T.image_count) AS i_count
            FROM
                (
                SELECT continent, date_added,
                        (
                            COALESCE((CASE WHEN image_1 IS NOT NULL THEN 1 ELSE 0 END), 0)
                            + COALESCE((CASE WHEN image_2 IS NOT NULL THEN 1 ELSE 0 END), 0)
                            + COALESCE((CASE WHEN image_3 IS NOT NULL THEN 1 ELSE 0 END), 0)
                            ) AS image_count
                        FROM `archdaily_dataset.table-post-id`
                        ) as T
            GROUP BY T.continent
            HAVING continent IS NOT NULL
            """
        )
    df2 = client.query(query_current_model).to_dataframe()
    df2['percentage'] = df2['i_count'].astype(str) + " (" + (df2['i_count'] / df2['i_count'].sum()).apply(lambda x: '{:,.1%}'.format(x)).astype(str) + ")"
    total_images2 = df2['i_count'].sum()

    
    data1 = go.Bar(
            x=df['i_count'],
            y=df['continent'],
            orientation = 'h',
            text = df['percentage'],
            )
    fig1 = go.Figure(data=data1)
    fig1.update_yaxes(
        categoryorder='total ascending'
    )
    fig1.update_layout(title_x=0.5,
                       title = {
                           'text':'Initial Dataset',
                           'font.size': 24}, 
                       xaxis_title="Number of Images"
                       )

    data2 = go.Bar(
            x=df2['i_count'],
            y=df2['continent'],
            orientation = 'h',
            text = df2['percentage'],
            )
    fig2 = go.Figure(data=data2)
    fig2.update_yaxes(
        categoryorder='total ascending'
    )
    fig2.update_layout(title_x=0.5,
                       title = {
                           'text':'Current Dataset',
                           'font.size': 24}, 
                       xaxis_title="Number of Images"
                       )
    
    st.markdown("<h3 style='text-align: center;'> Distribution of Classes in Initial vs. Current Dataset</h3>", unsafe_allow_html=True)
    
    col1,col2 = st.columns([1,1])
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown(f"""<div style='text-align: center;'><strong style="font-size:24px;">{total_images1:,.0f}</strong>   total images</div>""", unsafe_allow_html=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(f"""<div style='text-align: center;'><strong style="font-size:24px;">{total_images2:,.0f}</strong>   total images</div>""", unsafe_allow_html=True)

def get_prediction(file):
    
    # import pre- trained model 
    model = torch.load('entire_resnet50_model.pth')
    loader = Compose([Resize((256,256)), ToTensor()])
    
    image = Image.open(file)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet

    with torch.no_grad():
        model.eval()
        prediction = model(image)
    prob = nnf.softmax(prediction, dim=1)
    prediction_array = prob[0].numpy()
    
    return prediction_array

def predict_image():
    
    # st.markdown(f"<h3 style='text-align: center;'> Given a wait time of {user_option}:</h3>", unsafe_allow_html=True)
    
    col1,col2,col3 = st.columns([.3,1,.3])
    with col2:
        file = st.file_uploader("Upload a image file", type=["jpg", "jpeg", "png"])
        if file is not None:
            st.image(file, use_column_width=True)
            
            #get prediction for image
            prediction_array = get_prediction(file)
            
            st.markdown("<h4 style='text-align: center;'> The model predicted the following: </h4>", unsafe_allow_html=True)
            
    col1, col2, col3,col4,col5,col6,col7,col8 = st.columns(8)
    col1.metric("Africa", f"{prediction_array[0]:,.0%}")
    col2.metric("Asia", f"{prediction_array[1]:,.0%}")
    col3.metric("Australia", f"{prediction_array[2]:,.0%}")
    col4.metric("Central America", f"{prediction_array[3]:,.0%}")
    col5.metric("Europe", f"{prediction_array[4]:,.0%}")
    col6.metric("Middle East", f"{prediction_array[5]:,.0%}")
    col7.metric("North America", f"{prediction_array[6]:,.0%}")
    col8.metric("South America", f"{prediction_array[7]:,.0%}")
## functions end here, title, sidebar setting and descriptions start here

st.markdown("<h1 style='text-align: center;'>Image Classifier</h1>",unsafe_allow_html=True)
st.write("")
st.markdown("""A post on Reddit can contain zero or more comments. We can view comments as a stream of data over time (a time series).
            The plot below shows the total number of comments per hour for the entire **r/dataengineering** subreddit (a subreddit is a collection of posts).""")

dataset_classes()

st.write('')
st.write('')
st.markdown("<h3 style='text-align: center;'> Image Prediction Endpoint </h3>", unsafe_allow_html=True)

predict_image()


with st.sidebar:
    
    with st.expander("Click to learn more about this dashboard"):
        st.markdown(f"""
        This dashboard is designed as a mock user endpoint for a data engineering project.
        Details about the data engineering pipeline can be found at [here](https://github.com/mchion/incremental_training).
        
        This is NOT a data science or machine learning project.
        
        *The image classification model here is re-trained in its entirety on a weekly basis with new images*

        """)
