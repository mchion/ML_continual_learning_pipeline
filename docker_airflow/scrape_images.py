from bs4 import BeautifulSoup
import requests
import pandas as pd
import pandas_gbq
import os

from datetime import datetime
import pytz
from google.cloud import pubsub_v1
from google.oauth2 import service_account
from google.cloud import bigquery

PAGES = 5
PROJECT_ID = "project-archdaily"
TOPIC_ID = "new_images"
TABLE_ID = "archdaily_dataset.table-post-id"
SERVICE_ACCOUNT_DICT = os.environ.get("GOOLE_SERVICE_ACCOUNT")

credentials = service_account.Credentials.from_service_account_info(SERVICE_ACCOUNT_DICT)
pandas_gbq.context.credentials = credentials

COUNTRY_MAPPING = dict(pandas_gbq.read_gbq("SELECT * FROM `project-archdaily.archdaily_dataset.table-country-mapping`", project_id=PROJECT_ID).values)


# def get_bigquery_connection():
#     '''Initialize and connect to the BigQuery database'''
    
#     client_bq = bigquery.Client(project=SERVICE_ACCOUNT_DICT['project_id'], credentials=credentials)
#     return client_bq

def scrape_primary_links(pages):

    '''Find new primary links'''
    all_projects_list =[]
    for page_number in range(1, pages+1):
        url = f"https://www.archdaily.com/page/{page_number}"
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")

        # Get all articles
        articles = soup.find_all(itemtype="http://schema.org/Article")

        # Retrieve only images from posts labeled as 'project' (filter by label "Save this project")
        # Archdaily has posts labeled as 'article" that are not country specific
        projects = [article for article in articles if article.select_one('.single-bookmark').select_one('.afd-save-item')
                    .get('data-message')=='Save this project']

        count = 0
        for project in projects:
            try:
                post_date = project.select_one('.date-publication').span.time.get('datetime')
                date_added = datetime.now().astimezone(pytz.timezone('UTC')).strftime('%Y-%m-%d')
                primary_link = "http://archdaily.com" + project.select_one('.js-image-size__link').get('href')
                post_id = primary_link.split('/')[3]
                location_text = project.select_one('.afd-specs__header-location').text

                if "," in location_text:
                    country = location_text.split(',')[-1].strip()
                else:
                    country = location_text
                continent = COUNTRY_MAPPING.get(country)

                project_info = (post_id, post_date, date_added, primary_link, country,continent)
                all_projects_list.append(project_info)
                count +=1
            except AttributeError:
                print(f'Skipped link on page {page_number} with count {count}')

    df_all = pd.DataFrame(all_projects_list, columns =['post_id', 'post_date', 'date_added', 'primary_link', 'country', 'continent'])

    return df_all

def remove_duplicates(df_all):

    # load table of existing links already retrieved in prior runs
    sql1 = """SELECT post_id
    FROM `project-archdaily.archdaily_dataset.table-post-id`
    """
    df_bigquery = pandas_gbq.read_gbq(sql1, project_id=PROJECT_ID)
    
    df_all['post_id'] = df_all['post_id'].astype('int64')

    # Outer join
    outer_df = df_all.merge(df_bigquery, how="outer", left_on='post_id', right_on='post_id', indicator=True)

    # Keep only the ones in left (df_combined) b/c we have already retrieved the ones on the right (df_csv)
    new_primary_links_df = outer_df[outer_df['_merge'] == 'left_only'].drop(columns=["_merge"])

    return new_primary_links_df


def send_to_pubsub(message,filename):

    publisher = pubsub_v1.PublisherClient(credentials=credentials)
    topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)

    future = publisher.publish(topic_path, message, name=filename)
    print(future.result())

    print(f"Published messages to {topic_path}.")


def save_images_locally(image, filename):

    date_today = datetime.now().astimezone(pytz.timezone('UTC')).strftime('%Y-%m-%d')
    folder = f'Continents-batch{date_today}' 

    if not os.path.exists(folder):
        os.makedirs(folder)

    label = filename.split('-')[1]
    labelFolder = os.path.join(folder, label)

    if not os.path.exists(labelFolder):
        os.makedirs(labelFolder)

    destination = os.path.join(labelFolder, filename)

    with open(destination,"wb") as f:
        f.write(image)
    

def scrape_secondary_images(primary_link, post_id, continent):

    # Scrape 3 images from the primary link pages
    page = requests.get(primary_link)
    soup = BeautifulSoup(page.content, "html.parser")

    # Retrieve main image
    try:
        main_link = [soup.picture.source.get('srcset')]
    except:
        main_link = []
    
    # Retrieve 2 sub-images
    try:
        images = soup.find_all("a", class_="js-image-size__link lazy-anchor")
        sub_links = [img.contents[0].get('data-src') 
                     for img in images
                     if img.contents[0].get('data-src') is not None][0:4]
    except:
        sub_links = []

    # Add sub_links to main img list
    img_links=list(dict.fromkeys(main_link + sub_links))
    
    # Loop through images and save them to appropriate continent folder
    file_image_name_list = []
    image_count = 1
    for j in img_links:
        filename = f"{post_id}-{continent}-image{image_count}.jpg"   #os.path.basename(j).split('?')[0]
        

        # retrieve and pickle image
        message = requests.get(j).content
        
        # save locally
        #save_images_locally(message, filename)
        
        # send image and filename to Google Pub/Sub
        send_to_pubsub(message,filename)
        
        file_image_name_list.append(filename)
        image_count+=1
        
        if image_count >=4:
            break

    return file_image_name_list

def send_info_to_bigquery(df_table): 
    schema=[
        {'name': 'post_id', 'type': "INTEGER"},
        {'name': 'post_date', 'type': "TIMESTAMP"},
        {'name': 'date_added', 'type': "TIMESTAMP"},
        {'name': 'primary_link', 'type': "STRING"},
        {'name': 'country', 'type': "STRING"},
        {'name': 'continent', 'type': "STRING"},
        {'name': 'image_1', 'type': "STRING"},
        {'name': 'image_2', 'type': "STRING"},
        {'name': 'image_3', 'type': "STRING"}
        ]
    pandas_gbq.to_gbq(df_table, TABLE_ID, project_id=PROJECT_ID,if_exists='append',table_schema=schema)

def import_from_list(df_table):

    df_table['post_date'] = pd.to_datetime(df_table['post_date'])
    df_table['date_added'] = pd.to_datetime(df_table['date_added'])
    
    for row in df_table.index:
        
        if pd.isnull(df_table.at[row,'continent']):
            file_image_name_list = []
        else:
            file_image_name_list = scrape_secondary_images(df_table.at[row,'primary_link'], df_table.at[row,'post_id'], df_table.at[row,'continent'])

        for i,val in enumerate(file_image_name_list,start=1):
            df_table.at[row,f'image_{i}'] = val
    
    # load new links to bigquery database
    send_info_to_bigquery(df_table)

def main():
    df_all = scrape_primary_links(PAGES)
    
    #client_bq = get_bigquery_connection()
    df_new = remove_duplicates(df_all)
    import_from_list(df_new)
    
    
if __name__ =='__main__':
    main()