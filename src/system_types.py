from sklearn.pipeline import Pipeline
from constants import without_pca_system, with_pca_system, with_pca_and_dwt_system

system_types = [without_pca_system, with_pca_system, with_pca_and_dwt_system]

def create_pipeline(system_type, **kwargs):
    if system_type == without_pca_system:
        normalizer = kwargs['normalizer']
        xgboost = kwargs['xgboost']
        steps = [('normalizer', normalizer), ('xgboost', xgboost)]
    
    if system_type == with_pca_system:
        normalizer = kwargs['normalizer']
        pca = kwargs['pca']
        xgboost = kwargs['xgboost']
        steps = [('normalizer', normalizer), ('pca', pca), ('xgboost', xgboost)]
    
    if system_type == with_pca_and_dwt_system:
        normalizer = kwargs['normalizer']
        pca = kwargs['pca']
        dwt = kwargs['dwt']
        xgboost = kwargs['xgboost']
        steps = [('normalizer', normalizer), ('pca', pca), ('dwt', dwt), ('xgboost', xgboost)]
    
    else:
        raise ValueError('Incorrect system type.')

    # Print the steps being applied in the pipeline
    print(f"Pipeline steps for {system_type}: {[step[0] for step in steps]}")
    return Pipeline(steps)
