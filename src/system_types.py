from sklearn.pipeline import Pipeline
from constants import without_pca_system, with_pca_system, with_pca_and_dwt_system

# system_types = [without_pca_system, with_pca_system, with_pca_and_dwt_system]
# temporarily
system_types = [without_pca_system, with_pca_system]


def create_pipeline(system_type, **kwargs):
    pipeline = None

    if system_type == without_pca_system:
        normalizer = kwargs['normalizer']

        pipeline = Pipeline([('normalizer', normalizer)])
    elif system_type == with_pca_system:
        normalizer = kwargs['normalizer']
        pca = kwargs['pca']

        pipeline = Pipeline(
            [
                ('normalizer', normalizer),
                ('pca', pca),
            ]
        )
    elif system_type == with_pca_and_dwt_system:
        normalizer = kwargs['normalizer']
        pca = kwargs['pca']
        dwt = kwargs['dwt']

        pipeline = Pipeline(
            [
                ('normalizer', normalizer),
                ('pca', pca),
                ('dwt', dwt),
            ]
        )
    else:
        raise 'Incorrect system type.'

    if kwargs.get('xgboost') is not None:
        pipeline.steps.append(('xgboost', kwargs['xgboost']))

    return pipeline
