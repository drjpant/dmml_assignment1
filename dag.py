from prefect import flow, task

@task
def say_hello(name):
    print(f"Hello {name}!")

@flow
def my_flow():
    say_hello("Dheeraj")

if __name__ == "__main__":
    my_flow()