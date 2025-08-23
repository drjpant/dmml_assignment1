from prefect import flow, task
import subprocess

@task
def run_script(script_name):
    print(f"Running {script_name}...")
    subprocess.run(["python", script_name], check=True)

@flow
def pipeline_flow():
    run_script("data_ingestion.py")
    run_script("data_validation.py")
    run_script("data_preparation.py")
    run_script("model_training.py")

if __name__ == "__main__":
    pipeline_flow.serve(
        name="sequential-scripts",
        cron="*/5 * * * *"   # optional: every 5 min
    )