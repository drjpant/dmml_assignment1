from prefect import flow, task
from datetime import timedelta

@task
def say_hello(name):
    print(f"Hello {name}!")

@flow
def my_flow():
    say_hello("Dheeraj")

if __name__ == "__main__":
    # Serve the flow with a schedule (every 5 minutes)
    my_flow.serve(
        name="hello-flow-deployment",
        cron="*/5 * * * *"   # Cron expression for every 5 minutes
    )