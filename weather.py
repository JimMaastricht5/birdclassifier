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
# Module checks local weather, used to adjust picture brightness for input to species model
# As OpenWeatherMap APIs need a valid API key to allow responses,
# This stands for both free and paid (pro) subscription plans.
# You can signup for a free API key on the OWM website
# Please notice that the free API subscription plan is subject to requests throttling.
from auth import (
    weather_key
)
import requests
from datetime import datetime

# set city name or default to Madison, WI
# boolean true if the sky is clear
# sunrise and sunset as local time and full json string
class City_Weather:
    def __init__(self):
        self.city = 'Madison,WI,USA'
        self.base_url = 'http://api.openweathermap.org/data/2.5/weather?q='
        self.full_url = self.base_url + self.city + '&appid=' + weather_key
        self.sunrise = datetime.now()
        self.sunset = datetime.now()
        self.isclear = True
        self.weather = ''
        self.temp = ''
        self.local_weather()

    def local_weather(self):
        try:  # handle open weather API outages
            response = requests.get(self.full_url)
            # grab sunrise and sundown epoch data, parse epoch and convert to date time
            fulljson = response.json()
            print(fulljson)
            self.sun = str(response.json()['sys'])  # find sun rise and sunset string
            self.weather = str(response.json()['weather'])  # find general weather string
            self.temp = str(response.json()['main'])  # find general temp info
            start = sun.find('sunrise') + 10
            self.sunrise = datetime.fromtimestamp(int(sun[start: start + 10]))
            start = sun.find('sunset') + 9
            self.sunset = datetime.fromtimestamp(int(sun[start: start + 10]))

            # determine cloud conditions and return true if clear
            start = weather.find('main') + 8
            self.skycondition = weather[start: start + 5]
            if self.skycondition == 'Clear':
                self.isclear = True
            else:
                self.isclear = False
        except:
            self.isclear = True
            self.sunrise = datetime.now()
            self.sunset = datetime.now()
        return

    # update weather conditions
    def update_conditions(self):
        self.local_weather()
        return


def main():
    spweather = City_Weather()
    spweather.update_conditions()
    print(spweather.isclear)
    print(spweather.sunrise.strftime('%H:%M:%S'))
    print(spweather.sunset.strftime('%H:%M:%S'))
    print(spweather.weather)
    print(spweather.temp)


# test function
if __name__ == '__main__':
    main()
