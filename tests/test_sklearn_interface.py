from sklearn.utils.estimator_checks import parametrize_with_checks

# Import concrete, public estimators only
from cgmm import ConditionalGMMRegressor


def _estimators_for_checks():
    # Default-constructible estimators with no required args
    return [
        ConditionalGMMRegressor(),
        # add more as they become public
    ]


@parametrize_with_checks(_estimators_for_checks())
def test_sklearn_compatibility(estimator, check):
    check(estimator)
