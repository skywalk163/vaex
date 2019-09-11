import vaex
import pytest
import numpy as np
import contextlib


@contextlib.contextmanager
def relax_sklearn_check():
    import sklearn.preprocessing.data
    import sklearn.preprocessing._encoders
    import sklearn.preprocessing.base
    modules = [sklearn.preprocessing.data, sklearn.preprocessing._encoders, sklearn.preprocessing.base]
    old_check_arrays = {module: getattr(module, 'check_array') for module in modules}
    for module in modules:
       module.check_array = lambda x, *args, **kwargs: x
    yield
    for module in modules:
       module.check_array = old_check_arrays[module]

@contextlib.contextmanager
def no_array_casting():
    vaex.dataframe._allow_array_casting = True
    yield
    vaex.dataframe._allow_array_casting = True

@pytest.fixture
def df():
    x = np.arange(10, dtype=np.float64)
    y = x**2
    df = vaex.from_arrays(x=x, y=y)
    return df

def test_zeros_like(df):
    z = np.zeros_like(df.x)
    assert z.tolist() == [0] * 10

def test_sklearn_min_max_scalar(df):
    from sklearn.preprocessing import MinMaxScaler
    with relax_sklearn_check(), no_array_casting():
        scaler = MinMaxScaler()
        scaler.fit(df)

        dft = scaler.transform(df)
        assert isinstance(dft, vaex.DataFrame)
    X = np.array(df)
    Xt = scaler.transform(X)
    assert np.all(Xt == np.array(dft))

def test_sklearn_standard_scaler(df):
    from sklearn.preprocessing import StandardScaler
    with relax_sklearn_check(), no_array_casting():
        scaler = StandardScaler()
        scaler.fit(df)

        dft = scaler.transform(df)
        assert isinstance(dft, vaex.DataFrame)
    X = np.array(df)
    Xt = scaler.transform(X)
    assert np.all(Xt == np.array(dft))

def test_sklearn_power_transformer(df):
    from sklearn.preprocessing import PowerTransformer
    with relax_sklearn_check(), no_array_casting():
        scaler = PowerTransformer(standardize=False)
        scaler.fit(df)

        dft = scaler.transform(df.copy())
        assert isinstance(dft, vaex.DataFrame)
    X = np.array(df)
    Xt = scaler.transform(X)
    assert np.all(Xt == np.array(dft))
