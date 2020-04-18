import pytest

from trftools.pipeline._model import DefinitionError, Model, StructuredModel, Comparison


EXPRESSION = 1
X1_V_X0 = 2


def test_model():
    xyz = Model.coerce('x + y + z')
    xy = Model.coerce('x + y')
    yz = Model.coerce('y + z')
    y = Model.coerce('y')
    z = Model.coerce('z')
    assert xy + z == xyz
    assert xyz - z == xy
    assert xy.intersection(yz) == y


models = {
    'x-abcd': 'x-a + x-b + x-c + x-d',
    'x-ab': 'x-a + x-b',
    'x-cd': 'x-c + x-d',
}
structured_models = {k: StructuredModel.coerce(v) for k, v in models.items()}

test_data = [
    # direct
    ('x + a > x + b', True, 'x + a', 'x + b'),
    ('x = x + y', True, 'x', 'x + y'),
    # omit
    ('x + y | y', True, 'x + y', 'x'),
    ('x + y | x', True, 'x + y', 'y'),
    ('x | x$shift', False, 'x', 'x$shift'),
    ('x + y + z | z$shift', False, 'x + y + z', 'x + y + z$shift'),
    ('x + y + z | y$shift + z$shift', False, 'x + y + z', 'x + y$shift + z$shift'),
    # add
    ('x +| y', True, 'x + y', 'x'),
    ('x + y +| z', True, 'x + y + z', 'x + y'),
    ('x +| y$permute', False, 'x + y', 'x + y$permute'),
    # named direct
    ('x-ab < x-cd', True, 'x-a + x-b', 'x-c + x-d'),
    ('x-ab < x-cd', False, 'x-a + x-b', 'x-c + x-d'),
    # named omit
    ('x-ab | x-a$shift', False, 'x-a + x-b', 'x-b + x-a$shift'),
    ('x-ab | x-b$shift', False, 'x-a + x-b', 'x-a + x-b$shift'),
    ('x-abcd | x-ab$shift', False, 'x-a + x-b + x-c + x-d', 'x-c + x-d + x-a$shift + x-b$shift'),
    ('x-ab | x-b', True, 'x-a + x-b', 'x-a'),
    ('x-abcd | x-ab', True, 'x-a + x-b + x-c + x-d', 'x-c + x-d'),
    # named add
    ('x-ab +| x-c', True, 'x-a + x-b + x-c', 'x-a + x-b'),
    ('x-ab +| x-c$shift', False, 'x-a + x-b + x-c', 'x-a + x-b + x-c$shift'),
    # named add2
    ('x-ab +| x-c > x-d', True, 'x-a + x-b + x-c', 'x-a + x-b + x-d'),
]
test_data = [(*t, None) if len(t) == 4 else t for t in test_data]


@pytest.mark.parametrize('string,cv,x1,x0,name', test_data, ids=[items[0] for items in test_data])
def test_comparison(string, cv, x1, x0, name):
    """Assert that comparison is parsed correctly"""
    if name is None:
        name = string
    elif name == '>':
        name = x1 + ' > ' + x0

    with pytest.raises(DefinitionError):
        Model.coerce(string)

    comparison = Comparison.coerce(string, cv, structured_models)

    assert isinstance(comparison, Comparison)
    assert comparison.x1.name == x1
    assert comparison.x0.name == x0
    assert comparison.name == name


def test_comparison_parser():
    with pytest.raises(ValueError):
        Comparison.coerce('model | whot$shift', False, structured_models)
