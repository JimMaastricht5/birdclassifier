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
from operator import itemgetter


class Census:
    def __init__(self):
        self.clear()
        self.census = []

    def clear(self):
        self.census = []

    # find index for visitor by name
    def find_visitor(self, visitor_name):
        loc = -1
        for i in range(len(self.census)):
            if self.census[i][0] == visitor_name:
                loc = i
                break
        return loc

    # find visitor by census name, increment count, and update time
    def visitor(self, visitor_name, time_of_visit):
        if visitor_name == '' or visitor_name == ' ':
            visitor_name = 'undetermined'
        vindex = self.find_visitor(visitor_name)
        if vindex >= 0:
            self.census[vindex][1] += 1
            self.census[vindex][2] = time_of_visit
        else:  # add
            self.census.append([visitor_name, 1, time_of_visit])
        return

    # return count of visitor by name
    def report_census(self, visitor_name):
        vindex = self.find_visitor(visitor_name)
        if vindex >= 0:
            visitor_count = self.census[vindex][1]
            last_seen = self.census[vindex][2]
        else:
            visitor_count = 0  # haven't seen this visitor by name
            last_seen = datetime(2021, 1, 1, 0, 0, 0)  # initialize to a time that is not today
        return visitor_count, last_seen

    # sort census by count
    def get_census_by_count(self):
        self.census.sort(key=itemgetter(1), reverse=True)
        return self.census


def main():
    observed_time = datetime.now()
    popdogcats = Census()

    popdogcats.visitor('dog', observed_time)
    print('should be one dog', popdogcats.report_census('dog'))
    print('should be zero cats', popdogcats.report_census('cat'))

    popdogcats.visitor('cat', observed_time)
    popdogcats.visitor('cat', observed_time)
    print('should be one dog', popdogcats.report_census('dog'))
    print('should be two cats', popdogcats.report_census('cat'))
    print(popdogcats.get_census_by_count())
    observed = popdogcats.get_census_by_count()  # print count from prior day
    try:
        print(f'top 3 birds yesterday #1 {observed[0][0:2]}, #2 {observed[1][0:2]}')
    except IndexError:
        print('unable to post observations')

    popdogcats.clear()  # clear count for new day
    return


# testing code
if __name__ == "__main__":
    main()
