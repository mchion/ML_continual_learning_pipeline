# ML Image Classification Model with Continual Learning Pipeline
Pipeline that incrementally updates a neural network using PyTorch, Airflow and MLflow. Docker and Terraform used for reproducability. Streamlit used to create dashboard.

![Pipeline Diagram](/images/archdaily_diagram.svg)

Architectural images are webscraped from [**archdaily.com**](archdaily.com) and assigned categorical region labels based on the image's country of origin (determined by archdaily). For this project, images are classified into one of eight regions of the world.
[**Click here to view the dashboard endpoint**](https://mchion-ml-continual-learning-pipe-dashboardstreamlit-app-yjuna8.streamlit.app/).


## Data Extraction

- **Webscraping images**: In general, webscraping images takes a lot of guesswork. Pages from previous years may have changed formatting. For the most part, the html structure was the same. 
- **Types of images**: There were 3 types of architectural images 
- **Imbalanced classes**: During the course of webscraping, the images being scraped were 

## Data Loading

- **Google PubSub**: In order to pull messages more effectively and use lower latency, google pub/sub was implemented to take in images and their corresposnding text. 


## Data Transformation
- **Preprocessing images**: had to change image formats to fit machine learning model. Images had to pickled. 


## Data Modeling
- **ResNet50 Model**: We used the ResNet50 pretrained model for faster modeling. ResNet50 was a good balance between performance and accuracy. 

## Data Automation
- **Airflow**: In order to maintain the constant updating of 
- **MLFlow**: To keep track of various versions of models, used MLFlow to help simplify the choosing process once new samples are processed.  

## Data Visualization

An dashboard endpoint was made so that users that can users can upload their own architectural images and see the results for the 

![Dashboard General](/images/dashboard.png)

## Unit Testing

Although not implemeneted, will need to incorporate some unit testing involving PyTest. 

## Futher Directions and Considerations

- **New Categorical Training**: I think the classification of images to continents was a naive attempt at classification. Perhaps a better one would be to see the type of architecture the 
