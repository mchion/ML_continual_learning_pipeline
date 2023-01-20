# ML Image Classification with a Continual Learning Pipeline

**\*\*CURRENTLY UNDER CONSTRUCTION****

Pipeline that incrementally updates a neural network using PyTorch, Airflow and MLflow. Docker and Terraform used for reproducability. Streamlit used to create dashboard.

![Pipeline Diagram](/images/archdaily_diagram.svg)

Architectural images are webscraped from [**archdaily.com**](https://archdaily.com) and assigned categorical region labels based on the image's country of origin (determined by archdaily). For this project, images are classified into one of eight regions of the world.
[**Click here to view the dashboard endpoint**](https://mchion-ml-continual-learning-pipe-dashboardstreamlit-app-yjuna8.streamlit.app/)


## Data Extraction (Creation of Streaming Data Source)
Images webscraped from archdaily.com are immediately published to Google Pub/Sub. In order to avoid duplicate images being published, the app checks when the last images were imported. 


- **Webscraping images**: In general, webscraping images takes a lot of trial and error to figure out what particular HTML code will deliver the right images. For instance, pages from previous years may have changed formatting.
- **Types of images**: In general, architectural images can be broken down into 3 categories for our purposes - outdoor, indoor, and plans. This was a bottleneck since images of indoors and plans tend to not be as distinctive as outdoor images of architecture. There was some manual removal of images  
- **Imbalanced classes**: During the course of webscraping, the images being scraped were found to be imbalanced. 


## Data Loading


- **Google BigQuery**: Metadata and file names for each image are stored in BigQuery. The database schema is as follows:

  | Column Name | Value | 
  | ------------ | --------- | 
  | post_id | ID of posting containing image |
  | post_date | date of posting containing image |
  | date_added | date/time webscraped |
  | primary_link | link to posting containing image |
  | country| country of image |
  | continent | continent of image |
  | image_1| image 1 name |
  | image_2 | image 2 name |
  | image_3 | image 3 name |
  

- **Google Pub/Sub**: In order to simulate consuming images from a streaming data source, we consumed images on our local servers from Google Pub/Sub. Attached to each image is a message containing the post ID, post timestamp, and image number.\
\
One note is that Google Pub/Sub is not particularly made for image sending, and if our image sizes were larger, we would have to use a more robust message queue like Kafka or RabbitMQ. And even though images were allowed to be sent unserialized to Google Pub/Sub, serialization - using something like Python Pickle - would have been needed if images were larger or sent through another service. 


## Data Preprocessing
Once images were loaded and saved locally, we resized them to a 256 x 256 format and normalized as a tensor image. But in order to resize the images, we had to first find the mean and standard deviation of the image set. 

```shell
def batch_mean_and_sd(loader):
    
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)        
    return mean,std
```

Once we found the mean and standard deviation of the set, we used these numbers to perform the necessary PyTorch transforms. As we can see below, the means for the image set were calculated to be `(0.5182, 0.5144, 0.4948)` and the standard deviations were calculated to be `(0.2526, 0.2517, 0.2865)`. 

```shell
transform = transforms.Compose([transforms.Resize((256,256)), 
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5182, 0.5144, 0.4948), (0.2526, 0.2517, 0.2865))])
```

## Data Modeling
- **ResNet50 Model**: We used the ResNet50 pretrained model for faster modeling. ResNet50 was chosen for a balance between performance and accuracy. 

## Data Automation
- **Airflow**: In order to automate the updates, airflow was used to cooridinate the consuming of images from Google Pub/Sub and updating the model.
- **MLFlow**: To keep track of various versions of models, used MLFlow to help simplify the choosing process once new samples are processed.  

## Data Visualization

A dashboard endpoint was made so that users that can users can upload their own architectural images and see what region the ML model predicts their image is from. This was pretty straightforward due to the ease of using [**Streamlit**](https://streamlit.io/). 

![Dashboard General](/images/dashboard.png)


## Futher Directions and Considerations

- **New Categories**: I think the classification of images to continents was a naive attempt at classification. Perhaps a better one would be to see the type of architecture the image represents - residential, commercial, entertainment, etc. **CURRENT EFFORTS ARE UNDERWAY TO CHANGE THIS**
