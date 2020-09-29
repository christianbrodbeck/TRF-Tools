import pytest

from trftools.pipeline._model import DefinitionError, Model, ModelExpression, StructuredModel, Comparison


EXPRESSION = 1
X1_V_X0 = 2

models = {
    'x-abcd': 'x-a + x-b + x-c + x-d',
    'x-ab': 'x-a + x-b',
    'x-cd': 'x-c + x-d',
    'xyz': 'x + y + z',
}
structured_models = {k: StructuredModel.coerce(v) for k, v in models.items()}


def test_model():
    xyz = Model.coerce('x + y + z')
    xy = Model.coerce('x + y')
    yz = Model.coerce('y + z')
    y = Model.coerce('y')
    z = Model.coerce('z')
    assert xy + z == xyz
    assert xyz - z == xy
    assert xy.intersection(yz) == y
    xy2 = ModelExpression.from_string("xyz - z").initialize(structured_models)
    assert xy2 == xy
    # duplicate term
    with pytest.raises(DefinitionError):
        Model.coerce("term-1 + term-2 + term-2")
    # i+s
    a = ModelExpression.from_string("xyz-i+s").initialize(structured_models)
    b = ModelExpression.from_string("x-i+s + y-i+s + z-i+s").initialize(structured_models)
    assert a.sorted_key == b.sorted_key
    c = ModelExpression.from_string("x + x-step + y + y-step + z + z-step").initialize(structured_models)
    assert a.sorted_key == c.sorted_key


test_data = [
    # direct
    ('x + a > x + b', True, 'x + a', 'x + b'),
    ('x = x + y', True, 'x', 'x + y'),
    ('x > 0', True, 'x', '0'),
    ('x + a > 0', True, 'x + a', '0'),
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
# allow name being different than args[0]
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
    comp = Comparison.coerce('x-abcd | x-ab = x-cd', named_models=structured_models)
    assert comp.x1.name == 'x-c + x-d'
    assert comp.x0.name == 'x-a + x-b'

    with pytest.raises(ValueError):
        Comparison.coerce('model | whot$shift', False, structured_models)
