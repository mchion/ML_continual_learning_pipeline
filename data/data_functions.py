import pickle
import os
from concurrent.futures import TimeoutError
from google.cloud import pubsub_v1
from datetime import datetime

def get_data_from_pubsub(project_id, subscription_id):

    # Number of seconds the subscriber should listen for messages
    timeout = 5.0

    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(project_id, subscription_id)

    # class Callable:
    #   def __init__(self, idx):
    #     self.idx = idx

    #   def callback(self, message_future):
    #     print(message_future)
    #     print(self.idx)

    def save_image(filename,image):

        date_today = datetime.now().astimezone(pytz.timezone('UTC')).strftime('%Y-%m-%d')
        folder = f'data_new/Continents-batch{date_today}' 

        if not os.path.exists(folder):
            os.makedirs(folder)

        label = filename.split('-')[1]
        labelFolder = os.path.join(folder, label)

        if not os.path.exists(labelFolder):
            os.makedirs(labelFolder)

        destination = os.path.join(labelFolder, filename)

        with open(destination,"wb") as f:
            f.write(image)

    def callback(message: pubsub_v1.subscriber.message.Message) -> None:
        print(f"Received {message}.")
        print
        if message.attributes:
            for key in message.attributes:
                value = message.attributes.get(key)
                save_image(value,message.data)
        message.ack()

    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    print(f"Listening for messages on {subscription_path}..\n")

    # Wrap subscriber in a 'with' block to automatically call close() when done.
    with subscriber:
        try:
            # When `timeout` is not set, result() will block indefinitely,
            # unless an exception is encountered first.
            streaming_pull_future.result(timeout=timeout)
        except TimeoutError:
            streaming_pull_future.cancel()  # Trigger the shutdown.
            streaming_pull_future.result()  # Block until the shutdown is complete.


