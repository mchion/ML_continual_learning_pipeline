import airflow
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator

from data_functions import get_data_from_pubsub
from models import preprocessing, update_model, data_to_archive


'''Google PubSub account names'''
PROJECT_ID = "project-archdaily"
SUBSCRIPTION_ID = 'new_images-sub'


BATCH_SIZE = 8
NUM_WORKERS = 9


with DAG(
    dag_id='update_DAG',
    default_args={
        'owner': 'airflow',
        'start_date': airflow.utils.dates.days_ago(1),
        'provide_context': True,
    },
	schedule_interval='@daily',
	catchup=False,
) as dag:
    t1 = PythonOperator(
        task_id='get_data_from_pubsub',
        python_callable=get_data_from_pubsub,
        op_kwargs={'project_id': PROJECT_ID,
                'subscription_id': SUBSCRIPTION_ID},
    )
    t2 = PythonOperator(
        task_id='preprocessing',
        python_callable=preprocessing,
        op_kwargs={'batch_size': BATCH_SIZE,
                'num_workers': NUM_WORKERS},
    )
    t3 = PythonOperator(
        task_id='update_model',
        python_callable=update_model,
        op_kwargs={'batch_size': BATCH_SIZE,
                'num_workers': NUM_WORKERS},
    )
    t4 = PythonOperator(
        task_id='data_to_archive',
        python_callable=data_to_archive,
    )

t1 >> t2 >> t3 >> t4