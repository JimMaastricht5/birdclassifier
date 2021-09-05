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
import weather
from datetime import datetime
from gpiozero import CPUTemperature
from subprocess import call


class DailyChores:

    def __init__(self, tweeter_obj, birdpop):
        self.curr_day = datetime.now().day
        self.curr_hr = datetime.now().hour
        self.starttime = datetime.now()
        self.weather_reported = False
        self.pop_reported = False
        self.tweeter = tweeter_obj
        self.cityweather = weather.City_Weather()  # init class and set var based on default of Madison WI
        self.birdpop = birdpop

    # end of process report
    def end_report(self):
        self.tweeter.post_status(f'Ending process at {datetime.now().strftime("%I:%M:%S %P")}.  Run time was ' +
                                 f'{divmod((datetime.now() - self.starttime).total_seconds(), 60)[0]} minutes')

    # check current cpu temp, print, shutdown if overheated
    def check_cpu_temp(self):
        cpu = CPUTemperature()
        print(f'***hourly temp check. cpu temp is: {cpu.temperature}C {(cpu.temperature * 9 / 5) + 32}F')
        try:
            if int(cpu.temperature) >= 86:  # limit is 85 C
                self.tweeter.post_status(f'***shut down. temp: {cpu.temperature}')
                call("sudo shutdown -poweroff")
        except:
            pass
        return

    # post weather conditions
    def weather_report(self):
        self.tweeter.post_status(f'current time and weather: {datetime.now().strftime("%I:%M:%S %P")}, ' +
                                 f'{self.cityweather.weatherdescription} ' +
                                 f'with {self.cityweather.skycondition}% cloud cover. Visibility of' +
                                 f'{self.cityweather.visibility} ft.' +
                                 f' Temp is currently {self.cityweather.temp}F with ' +
                                 f'wind speeds of {self.cityweather.windspeed} MPH.')
        return

    # post a report of the top population for birds
    # takes in a list of tuple objects
    # position 0 in tuple is bird name, 1 is count, 2 is last seen
    def top_pop_report(self):
        post_txt = ''  # force to string
        birdstr = ''  # used to force tuple to string
        observed = self.birdpop.get_census_by_count()
        post_txt = f'top 3 birds for day {str(self.curr_day)}'
        index = 0
        while index <= 2:  # top 3 skipping unknown species
            if observed[index][0:2] != '':  # skip the unknown species category
                birdstr = str(observed[index][0])  # grab top species name
                start = birdstr.find('(') + 1  # find start of common name, move one character to drop (
                end = birdstr.find(')')
                if start >= 0 and end >= 0:
                    cname = birdstr[start:end]
                else:
                    cname = birdstr
                birdstr = str(f', #{str(index + 1)} {cname} {observed[index][1]} ')  # top bird count & species name
                post_txt = post_txt + birdstr  # aggregate text for post
            index += 1
        self.tweeter.post_status(post_txt[0:150])  # grab full text up to 150 characters
        return

    # housekeeping for day and hour
    # takes a pointer to the population tracking object
    def hourly_and_daily(self):
        # post the weather once per day at 6am
        if self.weather_reported is False and self.cityweather.is_daytime() and \
                datetime.now().hour > 6 and datetime.now().minute >= 0:
            self.weather_report()
            self.weather_reported = True

        if self.pop_reported is False and \
                (self.cityweather.is_daytime is False or datetime.now().hour >= 22):
            self.pop_reported = True
            self.top_pop_report()
            self.birdpop.clear()  # clear count for new day

        if self.curr_hr != datetime.now().hour:  # check weather and CPU temp hourly
            self.check_cpu_temp()

        self.curr_hr = datetime.now().hour
        self.curr_day = datetime.now().day
        return
