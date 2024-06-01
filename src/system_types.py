from sklearn.pipeline import Pipeline

system_types = ['without_pca', 'with_pca']


def create_pipeline(system_type, **kwargs):
    match system_type:
        case 'without_pca':
            normalizer = kwargs['normalizer']
            xgboost = kwargs['xgboost']

            return Pipeline([('normalizer', normalizer), ('xgboost', xgboost)])
        case 'with_pca':
            normalizer = kwargs['normalizer']
            pca = kwargs['pca']
            xgboost = kwargs['xgboost']

            return Pipeline(
                [('normalizer', normalizer), ('pca', pca), ('xgboost', xgboost)]
            )
        case _:
            raise 'Incorrect system type.'
