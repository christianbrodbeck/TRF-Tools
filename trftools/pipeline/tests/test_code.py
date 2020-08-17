from trftools.pipeline import Code


def test_code():
    code = Code('stim~x-option$permute')
    assert code.code_with_rand == 'x-option$permute'
    assert code.stim == 'stim'
    assert code.code == 'x-option'
    assert code.shuffle_index is None
    assert code.shuffle == 'permute'
    assert code.shuffle_angle == 180

    code = Code('stim~x-option$[mask]permute')
    assert code.code_with_rand == 'x-option$[mask]permute'
    assert code.stim == 'stim'
    assert code.code == 'x-option'
    assert code.shuffle_index == 'mask'
    assert code.shuffle == 'permute'
    assert code.shuffle_angle == 180
