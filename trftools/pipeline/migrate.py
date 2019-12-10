# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Update file structure for pipeline changes"""
from collections import defaultdict
import os

from eelbrain._utils.basic import ask


def decim_to_samplingrate(e, verbose=True):
    raw_samplingrate = {}
    subject_srates = defaultdict(set)
    for (subject, _), srate in e._raw_samplingrate.items():
        subject_srates[subject].add(srate)
    for subject, srates in subject_srates.items():
        if len(srates) == 1:
            raw_samplingrate[subject] = srates.pop()
        else:
            raw_samplingrate[subject] = None

    # subject results
    rename = {}
    for path in e.glob('trf-file', True):
        properties = e._parse_trf_path(path)
        if 'decim' not in properties:
            continue
        raw_srate = raw_samplingrate[properties['subject']]
        trf_srate = raw_srate / int(properties['decim'])
        old = f' {properties["decim"]} '
        new = f' {trf_srate:g}Hz '
        assert path.count(old) == 1
        path_to = path.replace(old, new)
        rename[path] = path_to
        if verbose:
            print(path.replace(old, f'{old}->{new}'))

    if ask(f"Rename {len(rename)} files?", {'rename': 'Rename files'}, allow_empty=True):
        print('Renaming...')
        failed = {}
        for src, dst in rename.items():
            try:
                os.rename(src, dst)
            except OSError as err:
                failed[src] = err
        if failed:
            errors = {err.args for err in failed.values()}
            print(f"{len(failed)} failures:")
            for n, msg in errors:
                print(f'{n}: {msg}')
        print("Done")
    else:
        print('Abort')
