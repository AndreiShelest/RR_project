from sklearn.pipeline import Pipeline
from constants import without_pca_system, with_pca_system, with_pca_and_dwt_system

system_types = [without_pca_system, with_pca_system, with_pca_and_dwt_system]


def create_pipeline(system_type, **kwargs):
    if system_type == without_pca_system:
        normalizer = kwargs['normalizer']
        xgboost = kwargs['xgboost']

        return Pipeline([('normalizer', normalizer), ('xgboost', xgboost)])
    if system_type == with_pca_system:
        normalizer = kwargs['normalizer']
        pca = kwargs['pca']
        xgboost = kwargs['xgboost']

        return Pipeline(
            [
                ('normalizer', normalizer),
                ('pca', pca),
                ('xgboost', xgboost),
            ]
        )
    if system_type == with_pca_and_dwt_system:
        normalizer = kwargs['normalizer']
        pca = kwargs['pca']
        dwt = kwargs['dwt']
        xgboost = kwargs['xgboost']

        return Pipeline(
            [('normalizer', normalizer), ('pca', pca), ('dwt', dwt), ('xgboost', xgboost)]
        )

    raise 'Incorrect system type.'
