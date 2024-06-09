import mlflow
import pickle

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    print(data)
    model, dv = data
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment('mage_export')
    mlflow.sklearn.log_model(model, 'zoomcamp/sklearn/models')
    with open('./vectorizer.pickle', 'wb') as f:
        pickle.dump(dv, f)
    mlflow.log_artifact("./vectorizer.pickle", artifact_path="preprocessor")
 
