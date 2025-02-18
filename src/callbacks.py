import http.client

from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class NtfyCallback(TrainerCallback):
    """A callback class for sending notifications at the start and end of training.

    Args:
        topic (str): The Ntfy.sh topic to notify
    """

    topic: str

    def __init__(self, topic: str):
        self.topic = topic

    def on_train_begin(self, args, state, control, **kwargs):
        self.send_notification(f"Training Began: {args.output_dir}")

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        self.send_notification(f"Training Ended: {args.output_dir}")

    def send_notification(self, message):

        if len(self.topic) > 0 and self.topic[0] != "/":
            self.topic = "/" + self.topic

        try:
            # Define the URL and path
            url = "ntfy.sh"

            headers = {"Content-Type": "text/plain"}

            conn = http.client.HTTPSConnection(url)
            conn.request("POST", self.topic, message, headers)
            response = conn.getresponse()
            if response.status == 200:
                print("Notification sent successfully!")
            else:
                print(
                    f"Failed to send notification. Status code: {response.status}, Reason: {response.reason}"
                )
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Ensure the connection is closed
            conn.close()
