# MIT License
#
# 2024 Jim Maastricht
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
from typing import Union


def convert_to_list(input_str_list: Union[str, list]) -> list:
    """
    static function that checks if the input is a list or a string and converts the string to a list
    :param input_str_list: str or list
    :return: list
    """
    return input_str_list if isinstance(input_str_list, list) else [input_str_list]


def common_name(name: str) -> str:
    """
    pull the common name from the full name which contains species, common name, and sometimes sex
    :param name: full label or name for the species
    :return: string containing the common name such as Northern Cardinal
    """
    cname, sname = '', ''
    try:
        sname = str(name)
        sname = sname[sname.find(' ') + 1:] if sname.find(' ') >= 0 else sname  # remove index number
        sname = sname[0: sname.find('[') - 1] if sname.find('[') >= 0 else sname  # remove sex
        cname = sname[sname.find('(') + 1: sname.find(')')] if sname.find('(') >= 0 else sname  # common name
    except Exception as e:
        print(e)
    return cname
