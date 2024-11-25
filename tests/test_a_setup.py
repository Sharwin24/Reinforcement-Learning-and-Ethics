import filetype
import os
import re

n_free_response = 6


def test_setup():
    '''
    '''

    with open('netid', 'r') as inf:
        lines = inf.readlines()

    assert len(lines) == 1, "Just a single line with your NetID"

    netid = str(lines[0].strip())
    assert netid != "NETID_GOES_HERE", "Add your NetID"
    assert netid.lower() == netid, "Lowercase NetID, please"
    assert re.search(r"^[a-z]{3}[0-9]{3,4}$", netid) is not None, "Your NetID looks like xyz0123"

