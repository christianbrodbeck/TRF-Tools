from nose.tools import eq_, assert_raises

from trftools.pipeline._model import Model, parse_comparison


EXPRESSION = 1
X1_V_X0 = 2


def assert_parse(expression, x1, x0, name=EXPRESSION, named_models=None,
                 component_names={'x1': 'model-1', 'x0': 'model-0'}):
    """Assert that expression is parsed correctly

    Parameters
    ----------
    expression : str
        The comparison specification expression.
    x1 : str
        The ``x1`` model.
    x0 : str
        The ``x0`` model.
    name : EXPRESSION | X1_V_X0 | str
        The unique description of ``expression``.
    named_models : dict
        Named models.
    component_names : {str: str}
        Used to reconstrct the relative name when ``named_models`` is used.
    """
    if name is EXPRESSION:
        name = expression
    elif name is X1_V_X0:
        name = x1 + ' > ' + x0

    comparison = parse_comparison(expression, named_models)

    eq_(comparison.x1.name, x1)
    eq_(comparison.x0.name, x0)
    eq_(comparison.name, name)
    if named_models:
        eq_(comparison.relative_name(component_names), expression)


def test_parser():
    assert_parse('audspec ( > $rand)', 'audspec', 'audspec$rand', EXPRESSION)
    assert_parse('audspec + (a > b)', 'audspec + a', 'audspec + b')
    assert_parse('audspec- (a + b > a$rand + b$rand)',
                 'audspec-a + audspec-b', 'audspec-a$rand + audspec-b$rand', EXPRESSION)
    assert_parse('audspec- (a + b > a$rand + b$rand)',
                 'audspec-a + audspec-b', 'audspec-a$rand + audspec-b$rand', EXPRESSION)
    # empty x0_only
    assert_parse('audspec ( > + -b)', 'audspec', 'audspec + audspec-b')
    # x1 or x0 == common_base
    assert_parse('audspec > audspec + b',  'audspec', 'audspec + b', 'audspec + ( > b)')
    assert_parse('audspec + ( > b)', 'audspec', 'audspec + b')
    assert_parse('audspec ( + -a > -b)', 'audspec + audspec-a', 'audspec-b')
    assert_parse('audspec (-a > + -b)', 'audspec-a', 'audspec + audspec-b')
    # readable name
    assert_parse('word-a = word-b',
                 'word-a', 'word-b', 'word- (a = b)')
    assert_parse('word-b + word-a | word-b$rand',
                 'word-b + word-a', 'word-b$rand + word-a',
                 'word-a + word-b ( > $rand)')
    assert_parse('word-b + word-a | word-a$rand',
                 'word-b + word-a', 'word-b + word-a$rand',
                 'word-b + word-a ( > $rand)')

    named_models = {
        'model': Model('word-a + word-b + word-c + word-d'),
        'model-1': Model('word-a + word-b'),
        'model-0': Model('word-c + word-d'),
    }
    assert_parse('model-1 | word-a$rand',
                 'word-a + word-b', 'word-a$rand + word-b',
                 'word-b + word-a ( > $rand)', named_models)
    assert_parse('model-1 | word-a$rand + word-b$rand',
                 'word-a + word-b', 'word-a$rand + word-b$rand',
                 'word- (a + b > a$rand + b$rand)', named_models)
    assert_parse('model-1 | word-a > word-c',
                 'word-a + word-b', 'word-b + word-c',
                 'word-b + word- (a > c)', named_models)
    assert_parse('model-1 + word-c | word-c$rand',
                 'word-a + word-b + word-c', 'word-a + word-b + word-c$rand',
                 'word-a + word-b + word-c ( > $rand)', named_models)
    assert_parse('model-1 + word-c | word-a$rand',
                 'word-a + word-b + word-c', 'word-a$rand + word-b + word-c',
                 'word-b + word-c + word-a ( > $rand)', named_models)
    assert_parse('model-1 = model-0',
                 'word-a + word-b', 'word-c + word-d',
                 'word- (a + b = c + d)', named_models)
    assert_parse('model | model-0$rand',
                 'word-a + word-b + word-c + word-d',
                 'word-a + word-b + word-c$rand + word-d$rand',
                 'word-a + word-b + word- (c + d > c$rand + d$rand)',
                 named_models, {'x1': 'model', 'x0rand': 'model-0'})

    assert_raises(ValueError, parse_comparison, 'model | whot$rand', named_models)
