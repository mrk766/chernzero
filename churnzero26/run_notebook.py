import nbformat
from nbclient import NotebookClient
import sys

print("Executing churn_prediction.ipynb...")

try:
    with open('churn_prediction.ipynb', 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    client = NotebookClient(nb, timeout=600, kernel_name='python3')
    client.execute()
    
    with open('churn_prediction_executed.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print("churn_prediction_executed.ipynb generated successfully!")
except Exception as e:
    print(f"Error during notebook execution: {e}")
    sys.exit(1)
