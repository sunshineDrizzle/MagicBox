import warnings


def _filling(string, fillers_list, fillers, strings):
    """
    do filling

    Parameters:
    -----------
    string: string
        A string that may contain several placeholders {}.
    fillers_list: list
        Each element is a collection of fillers named fillers_c. Each fillers_c works on
        its corresponding placeholder {} in the 'string'. And there is an one-to-one
        correspondence in order between fillers_c and placeholders, which can also be specified
        by position parameters. Fillers in these fillers_cs are used to fill corresponding placeholders
        to generate strings we need.
        The number of generated string is equal to
        len(fillers_list[0]) * len(fillers_list[1]) * ... * len(fillers_list[-1])
    fillers: list
        There are some fillers used to fill the 'string' at one iteration in this list.
    strings: list
        Modified in situ to store generated strings.

    """
    idx = len(fillers)
    if idx != len(fillers_list):
        for filler in fillers_list[idx]:
            _filling(string, fillers_list, fillers + [filler], strings)
    else:
        strings.append(string.format(*fillers))


def get_strings_by_filling(string, fillers_list):
    """
    get strings by filling

    Parameters:
    -----------
    string: string
        A string that may contain several placeholders {}.
    fillers_list: list
        Each element is a collection of fillers named fillers_c. Each fillers_c works on
        its corresponding placeholder {} in the 'string'. And there is an one-to-one
        correspondence in order between fillers_c and placeholders, which can also be specified
        by position parameters. Fillers in these fillers_cs are used to fill corresponding placeholders
        to generate strings we need.
        The number of generated string is equal to
        len(fillers_list[0]) * len(fillers_list[1]) * ... * len(fillers_list[-1])
    """
    strings = []
    _filling(string, fillers_list, [], strings)

    return strings


def excel_col_label2number(col_lbl):
    """
    Convert Excel column label to number
    For example:
    'A' -> 1
    'B' -> 2
    'AA' -> 27
    'BB' -> 54
    Analogous to 10 in the decimalism, here is 26.

    Args:
        col_lbl (str): Excel column label

    Returns:
        int: Excel column number
    """
    if col_lbl != col_lbl.upper():
        warnings.warn(f"All letters of '{col_lbl}' will "
                      'be converted to uppercase at first.')
        col_lbl = col_lbl.upper()

    index = 0
    onset = ord('A') - 1
    for c in col_lbl:
        index = 26 * index + ord(c) - onset
    return index
