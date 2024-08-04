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
# population census, module keeps track of occurrences of an item along
# with the date time it was last counted (last seen)
from datetime import datetime
from collections import defaultdict
import static_functions
from typing_extensions import Union


def default_value() -> tuple:
    """
    default dictionary returns a tuple of zero count and the current date and time as the last seen value
    :return:
    """
    return 0, datetime.now()


class Census:
    """
    Census object tracks by key the count and last count of an item
    Reports out values upon request
    """
    def __init__(self) -> None:
        """
        Set up object with dictionaries for count and last occurrence
        :return: None
        """
        self.census_dict = defaultdict(default_value)
        self.census_occurrence = []
        self.first_time_seen = False
        return

    def clear(self) -> None:
        """
        clear dictionaries
        :return: none
        """
        self.census_dict = []  # clear it and re-establish
        self.census_dict = defaultdict(default_value)

    def visitors(self, visitor_names: Union[list, str], time_of_visit: datetime = datetime.now()) -> bool:
        """
        find visitor by census name, increment count, and update time
        make sure visitor names is a list and not a string
        :param visitor_names: list of names
        :param time_of_visit: datetime of occurrence
        :return: returns true if this is the first time the object was counted (first seen)
        """
        self.first_time_seen = False
        visitor_name_list = visitor_names if isinstance(visitor_names, list) else\
            static_functions.convert_to_list(visitor_names)
        for i, visitor_name in enumerate(visitor_name_list):
            print(visitor_name)
            if isinstance(visitor_name, str) and visitor_name.rstrip() != '':  # do we have a name?
                if self.census_dict[visitor_name][0] == 0:  # check census to see count for this species
                    self.first_time_seen = True
                self.census_dict[visitor_name] = (self.census_dict[visitor_name][0] + 1, time_of_visit)
                self.census_occurrence.append((visitor_name, time_of_visit.strftime("%Y-%m-%d %H:%M:%S")))
        return self.first_time_seen

    def report_census(self, visitor_names: Union[list, str]) -> dict:
        """
        return count of visitors by name along with last seen date time
        :param visitor_names: key name to look up aka dog or [dog, cat]
        :return:dictionary containing the subset of requested visitor counts
        """
        visitor_name_list = static_functions.convert_to_list(visitor_names)
        census_subset = {key: self.census_dict[key] for key in visitor_name_list}
        return census_subset

    def report_single_census_count(self, visitor_name: str) -> int:
        """
        return count of visitors by name
        :param visitor_name (key name) aka cat
        :return: int count
        """
        return self.census_dict[visitor_name][0]

    def get_census_by_count(self) -> dict:
        """
        sort census by count
        :return: sorted dictionary
        """
        return dict(sorted(self.census_dict.items(), key=lambda k_v: k_v[1][0], reverse=True))

    def get_occurrences(self) -> list:
        """
        returns a list of tuples containing the key and last seen or last counted time
        :return: list of tuples.  tuples is a single set of key / datetime values
        """
        return self.census_occurrence


def main():
    observed_time = datetime.now()
    popdogcats = Census()

    popdogcats.visitors('dog', observed_time)
    print('should be one dog', popdogcats.report_census('dog'))
    print('should be zero cats', popdogcats.report_census('cat'))

    popdogcats.visitors('cat', observed_time)
    popdogcats.visitors('cat', observed_time)
    print('should be one dog', popdogcats.report_census('dog'))
    print('should be two cats', popdogcats.report_census('cat'))
    print(popdogcats.report_single_census_count('cat'))
    popdogcats.visitors(['dog', 'cat', 'bird'], datetime.now())

    print('should be two dogs, three cats, and a bird')
    observed = popdogcats.get_census_by_count()  # print count from prior day
    print(observed)
    print('occurences:')
    print(popdogcats.get_occurrences())

    # mirror daily chorses reporting for testing
    # def short_name(birdname):
    #     start = birdname.find('(')
    #     end = birdname.find(')')
    #     return birdname[start + 1:end] if start >= 0 and end >= 0 else birdname

    post_txt = ''
    for index, birdkey in enumerate(observed):  # bird pop is list of tuples with 0th item species name
        # top count & species name
        birdstr = str(f'#{str(index + 1)}: {observed[birdkey][0]} {static_functions.common_name(birdkey)}, ')
        post_txt = post_txt + birdstr  # aggregate text for post
    print(post_txt)

    popdogcats.clear()  # clear count for new day
    return


# testing code
if __name__ == "__main__":
    main()
