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
# sunrise and sunset as local time
# all other variables returned as string type
class City_Weather:
    def __init__(self):
        # set defaults
        self.city = 'Madison,WI,USA'
        self.base_url = 'http://api.openweathermap.org/data/2.5/weather?q='
        self.units = 'Imperial'
        self.full_url = self.base_url + self.city + '&units=' + self.units + '&appid=' + weather_key
        self.cloudythreshold = 60

        # init variables
        self.sunrise = datetime.now()
        self.sunset = datetime.now()
        self.isclear = True
        self.weather = ''
        self.temp = ''
        self.local_weather()

    # grab sunrise and sundown epoch data, parse epoch and convert to date time
    def local_weather(self):
        try:  # handle open weather API outages
            response = requests.get(self.full_url)
            self.fulljson = response.json()
            # print(self.fulljson)
            self.weather = str(response.json()['weather'])  # find general weather string
            self.weatherdescription = str(response.json()['weather'][0]['description'])
            self.temp = str(response.json()['main']['temp'])
            self.pressure = str(response.json()['main']['pressure'])
            self.humidity = str(response.json()['main']['humidity'])
            self.visibility = str(response.json()['visibility'])
            self.windspeed = str(response.json()['wind']['speed'])
            self.sunrise = datetime.fromtimestamp(int(response.json()['sys']['sunrise']))
            self.sunset = datetime.fromtimestamp(int(response.json()['sys']['sunset']))
            self.skycondition = int(response.json()['clouds']['all'])  # % cloudy
            if self.skycondition < self.cloudythreshold:
                self.isclear = True
            else:
                self.isclear = False
        except:
            print('weather error encountered')
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
    print(spweather.temp)
    print(spweather.weatherdescription)
    print(spweather.windspeed)
    print(spweather.pressure)
    print(spweather.humidity)
    print(spweather.visibility)


# test function
if __name__ == '__main__':
    main()
