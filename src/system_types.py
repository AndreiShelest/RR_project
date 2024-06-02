from sklearn.pipeline import Pipeline
from constants import without_pca_system, with_pca_and_dwt_system

system_types = [without_pca_system, with_pca_and_dwt_system]


def create_pipeline(system_type, **kwargs):
    if system_type == without_pca_system:
        normalizer = kwargs['normalizer']
        mooga = kwargs['mooga']

        return Pipeline([('normalizer', normalizer), ('mooga', mooga)])
    if system_type == with_pca_and_dwt_system:
        normalizer = kwargs['normalizer']
        pca = kwargs['pca']
        dwt = kwargs['dwt']
        mooga = kwargs['mooga']

        return Pipeline(
            [('normalizer', normalizer), ('pca', pca), ('dwt', dwt), ('mooga', mooga)]
        )

    raise 'Incorrect system type.'
