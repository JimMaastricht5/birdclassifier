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
# motion detector with builtin bird detection and bird classification
# built by JimMaastricht5@gmail.com
from datetime import datetime
from gpiozero import CPUTemperature
import time
import static_functions


# find common / short name in species label
# def short_name(birdname):
#     start = birdname.find('(')
#     end = birdname.find(')')
#     return birdname[start + 1:end] if start >= 0 and end >= 0 else birdname


class DailyChores:
    """
    Class handles hourly and daily chores including checking the temp on the CPU,
    morning weather report, evening report on bird species observed
    takes a collection of objects as input tweeter.py, population.py, output_steam.py
    coordinates calls across these objects to complete work on schedule
    """
    def __init__(self, tweeter_obj, birdpop, city_weather, output_class=None, maxcpu_c_temp: float = 86) -> None:
        """
        setup class to support chores
        :param tweeter_obj: TweeterClass object form tweeter.py
        :param birdpop: Census object from population.py
        :param city_weather: CityWeather object from weather.py
        :param output_class: Controller object from output_stream.py
        :param maxcpu_c_temp: float containing max temp in C, if exceeded sleep until it cools off
        return: none
        """
        self.curr_day = datetime.now().day
        self.curr_hr = datetime.now().hour
        self.starttime = datetime.now()
        self.weather_reported = False
        self.pop_reported = False
        self.tweeter = tweeter_obj
        self.cityweather = city_weather
        self.birdpop = birdpop
        self.output_class = output_class  # take an argument of class Controller from output_stream.py
        self.output_func = output_class.message if output_class is not None else print
        self.maxcpu_c_temp = maxcpu_c_temp
        return

    # end of process report
    def end_report(self) -> None:
        """
        Send ending process final message to twitter
        :return: none
        """
        self.tweeter.post_status(f'Ending process at {datetime.now().strftime("%I:%M:%S %P")}.  Run time was ' +
                                 f'{divmod((datetime.now() - self.starttime).total_seconds(), 60)[0]} minutes')
        return

    # check current cpu temp, print, shutdown if overheated
    def check_cpu_temp(self) -> None:
        """
        check tem and sleep if too hot
        :return: none
        """
        cpu = CPUTemperature()
        self.output_func(f'***hourly temp check. cpu temp is: {cpu.temperature:.1f}C'
                         f' {(cpu.temperature * 9 / 5) + 32:.1f}F')
        try:
            if int(cpu.temperature) >= self.maxcpu_c_temp:  # limit is 85 C
                self.tweeter.post_status(f'***sleeping for 30 minutes. temp: {cpu.temperature}C'
                                         f' {(cpu.temperature * 9 / 5) + 32:.1f}F')
                time.sleep(1800)  # 1800 seconds is 30 min, allow CPU to cool
        except Exception as e:
            self.output_func('Error in temp shutdown protection:', e)
            pass  # uncharted territory....
        return

    # post weather conditions
    def weather_report(self) -> None:
        """
        write a weather report to twitter, lets everyone know the detectors is active
        :return: none
        """
        self.tweeter.post_status(f'Current time and weather for {self.cityweather.city} '
                                 f'{datetime.now().strftime("%I:%M:%S %P")}, ' +
                                 f'{self.cityweather.weatherdescription} ' +
                                 f'with {self.cityweather.skycondition}% cloud cover. Visibility of ' +
                                 f'{self.cityweather.visibility} ft.' +
                                 f' Temp is currently {self.cityweather.temp}F with ' +
                                 f'wind speeds of {self.cityweather.windspeed} MPH.' +
                                 f'Sunrise was at {self.cityweather.sunrise: %H:%m}. ' +
                                 f'Sunset is at {self.cityweather.sunset: %H:%m}.')
        return

    def top_pop_report(self) -> None:
        """
        Post a report of the top population for birds.  Takes in a list of tuple objects
        position 0 in tuple is bird name, 1 is count, 2 is last seen time
        :return: none
        """
        try:
            observed = self.birdpop.get_census_by_count()
            post_txt = f'Top birds for day {str(self.curr_day)} - '
            for index, birdkey in enumerate(observed):  # bird pop is list of tuples with 0th item species name
                bird_txt = str(f'#{str(index + 1)}: {observed[birdkey][0]} {static_functions.common_name(birdkey)}, ') \
                    if observed[birdkey][0] > 0 else ''  # post observed bird if count > 0 else keep prior txt
                post_txt = post_txt + bird_txt if len(bird_txt) + len(post_txt) < 280 else post_txt
            self.tweeter.post_status(post_txt[0:279])  # grab full text up to 280 characters
        except Exception as e:
            self.output_func('Error in daily population report:', e)
            pass  # just keep going...
        return

    def hourly_and_daily(self, filename: str = '', report_pop: bool = False) -> None:
        """
        housekeeping for day and hour. handles weather report, population report, and cpu check
        :param filename: file name to write out sample image to, used to check seed levels on web messages
        :param report_pop: boolean telling the class to reports the daily population if true
        :return: none
        """
        # post the weather once during the daytime and while the sun is rising
        if self.weather_reported is False and self.cityweather.is_daytime() and \
                self.cityweather.is_dawn():
            self.weather_report()
            self.weather_reported = True

        if report_pop:  # process is ending report populations observations
            self.pop_reported = True
            self.top_pop_report()
            self.birdpop.clear()  # clear count for new day

        if self.curr_hr != datetime.now().hour:  # check weather and CPU temp hourly
            self.check_cpu_temp()
            if self.output_class is not None:  # output is not being sent to the default print function
                self.output_class.occurrences(self.birdpop.get_occurrences())
                self.output_class.message(message=f'Seed and camera position check. ', image_name=filename)
                self.output_class.flush()

        self.curr_hr = datetime.now().hour
        self.curr_day = datetime.now().day
        return
