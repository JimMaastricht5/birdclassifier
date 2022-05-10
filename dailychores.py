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
# motion detector with builtin bird detection and bird classification
# built by JimMaastricht5@gmail.com
# class takes a tweeter.py class and population.py class as input on init
# import weather
from datetime import datetime
from gpiozero import CPUTemperature
from subprocess import call


# find common / short name in species label
def short_name(birdname):
    start = birdname.find('(')
    end = birdname.find(')')
    return birdname[start + 1:end] if start >= 0 and end >= 0 else birdname


class DailyChores:

    def __init__(self, tweeter_obj, birdpop, city_weather, output=print):
        self.curr_day = datetime.now().day
        self.curr_hr = datetime.now().hour
        self.starttime = datetime.now()
        self.weather_reported = False
        self.pop_reported = False
        self.tweeter = tweeter_obj
        self.cityweather = city_weather
        self.birdpop = birdpop
        self.output = output  # set print function or custom class

    # end of process report
    def end_report(self):
        self.tweeter.post_status(f'Ending process at {datetime.now().strftime("%I:%M:%S %P")}.  Run time was ' +
                                 f'{divmod((datetime.now() - self.starttime).total_seconds(), 60)[0]} minutes')

    # check current cpu temp, print, shutdown if overheated
    def check_cpu_temp(self):
        cpu = CPUTemperature()
        self.output(f'***hourly temp check. cpu temp is: {cpu.temperature}C {(cpu.temperature * 9 / 5) + 32}F')
        try:
            if int(cpu.temperature) >= 86:  # limit is 85 C
                self.tweeter.post_status(f'***shut down. temp: {cpu.temperature}')
                call("sudo shutdown -poweroff")
        except Exception as e:
            self.output('Error in temp shutdown protection:', e)
            pass  # uncharted territory....
        return

    # post weather conditions
    def weather_report(self):
        self.tweeter.post_status(f'current time and weather: {datetime.now().strftime("%I:%M:%S %P")}, ' +
                                 f'{self.cityweather.weatherdescription} ' +
                                 f'with {self.cityweather.skycondition}% cloud cover. Visibility of ' +
                                 f'{self.cityweather.visibility} ft.' +
                                 f' Temp is currently {self.cityweather.temp}F with ' +
                                 f'wind speeds of {self.cityweather.windspeed} MPH.')
        return

    # post a report of the top population for birds
    # takes in a list of tuple objects
    # position 0 in tuple is bird name, 1 is count, 2 is last seen
    def top_pop_report(self):
        try:
            observed = self.birdpop.get_census_by_count()
            post_txt = f'top birds for day {str(self.curr_day)}. '
            for index, birdkey in enumerate(observed):  # bird pop is list of tuples with 0th item species name
                post_txt = post_txt + str(f'#{str(index + 1)}: {observed[birdkey][0]} {short_name(birdkey)}, ') \
                    if observed[birdkey][0] > 0 else post_txt  # post observed bird if count > 0 else keep prior txt
            self.tweeter.post_status(post_txt[0:279])  # grab full text up to 280 characters
        except Exception as e:
            self.output('Error in daily population report:', e)
            pass  # just keep going...
        return

    # housekeeping for day and hour
    # takes a pointer to the population tracking object
    def hourly_and_daily(self, filename='', report_pop=False):
        # post the weather once per day at 6am
        if self.weather_reported is False and self.cityweather.is_daytime() and \
                datetime.now().hour > 6 and datetime.now().minute >= 0 and \
                datetime.now().hour < 8:
            self.weather_report()
            if filename != '':  # if an img file name was passed post it
                self.tweeter.post_image_from_file(message=f'Morning seed and camera position check. ',
                                                  file_name=filename)
            self.weather_reported = True

        if report_pop:  # process is ending report populations observations
            self.pop_reported = True
            self.top_pop_report()
            self.birdpop.clear()  # clear count for new day

        if self.curr_hr != datetime.now().hour:  # check weather and CPU temp hourly
            self.check_cpu_temp()
            self.output.occurrences(self.birdpop.get_census_by_count())

        self.curr_hr = datetime.now().hour
        self.curr_day = datetime.now().day
        return
