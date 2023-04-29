import os.path

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(ROOT_PATH, "results")
os.makedirs(RESULTS_PATH, exist_ok=True)
