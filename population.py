# MIT License
#
# 2021 Jim Maastricht
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
# population census
#
from datetime import datetime
from collections import defaultdict


# default dictionary returns a tuple of zero count and the current date and time as last seen
def default_value():
    return 0, datetime.now()


class Census:
    def __init__(self):
        self.census_dict = defaultdict(default_value)
        self.first_time_seen = False

    def clear(self):
        self.census_dict = []  # clear it and re-establish
        self.census_dict = defaultdict(default_value)

    # find visitor by census name, increment count, and update time
    def visitors(self, visitor_names, time_of_visit=datetime.now()):
        visitor_name_list = []
        self.first_time_seen = False
        if type(visitor_names) != list:
            visitor_name_list.append(visitor_names)
        else:
            visitor_name_list = visitor_names
        for i, visitor_name in enumerate(visitor_name_list):
            visitor_name = visitor_name if visitor_name != '' and visitor_name != ' ' else 'undetermined'
            if self.census_dict[visitor_name][0] == 0:
                self.first_time_seen = True
            self.census_dict[visitor_name] = (self.census_dict[visitor_name][0] + 1, time_of_visit)
        return

    # return count of visitors by name along with last seen date time
    def report_census(self, visitor_names):
        visitor_name_list = []
        if type(visitor_names) != list:
            visitor_name_list.append(visitor_names)
        else:
            visitor_name_list = visitor_names
        census_subset = {key: self.census_dict[key] for key in visitor_name_list}
        return census_subset

    # sort census by count
    def get_census_by_count(self):
        return dict(sorted(self.census_dict.items(), key=lambda k_v: k_v[1][0], reverse=True))


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
    popdogcats.visitors(['dog', 'cat', 'bird'], datetime.now())

    print('should be two dogs, three cats, and a bird')
    observed = popdogcats.get_census_by_count()  # print count from prior day
    print(observed)

    # mirror daily chorses reporting for tesitng
    def short_name(birdname):
        start = birdname.find('(')
        end = birdname.find(')')
        return birdname[start + 1:end] if start >= 0 and end >= 0 else birdname

    post_txt = ''
    for index, birdkey in enumerate(observed):  # bird pop is list of tuples with 0th item species name
        birdstr = str(f'#{str(index + 1)}: {observed[birdkey][0]} {short_name(birdkey)}, ')  # top count & species name
        post_txt = post_txt + birdstr  # aggregate text for post
    print(post_txt)

    popdogcats.clear()  # clear count for new day
    return


# testing code
if __name__ == "__main__":
    main()
