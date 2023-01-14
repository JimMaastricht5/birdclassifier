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
from datetime import timedelta
import time


# set city name or default to Madison, WI
# boolean true if the sky is clear
# sunrise and sunset as local time
# all other variables returned as string type
class CityWeather:
    def __init__(self, city='Madison,WI,USA', units='Imperial', iscloudy=60):
        # set defaults
        self.city = city
        self.base_url = 'http://api.openweathermap.org/data/2.5/weather?q='
        self.units = units
        self.full_url = self.base_url + self.city + '&units=' + self.units + '&appid=' + weather_key
        self.cloudythreshold = iscloudy

        # init variables
        self.sunrise = datetime.now()
        self.sunset = datetime.now()
        self.isclear = True
        self.weather = ''
        self.temp = ''
        self.visibility = ''
        self.windspeed = ''
        self.skycondition = 0
        self.weatherdescription = ''
        self.pressure = ''
        self.humidity = ''
        # call for weather and fill variables
        self.local_weather()

    # grab sunrise and sundown epoch data, parse epoch and convert to date time
    def local_weather(self):
        try:  # handle open weather API outages
            response = requests.get(self.full_url)
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
        except Exception as e:
            print('weather error encountered')
            print(e)
        return

    # is it daylight, now must be between sunrise and sunset
    def is_daytime(self):
        return datetime.now() > self.sunrise or datetime.now() < self.sunset

    def is_twilight(self):
        return self.is_dawn() or self.is_dusk()

    def is_dawn(self):
        from_sunrise_min = datetime.now() - self.sunrise
        from_sunrise_min = from_sunrise_min.total_seconds() / 60
        return from_sunrise_min < 60

    def is_dusk(self):
        from_sunset_min = self.sunset - datetime.now()
        from_sunset_min = from_sunset_min.total_seconds() / 60
        return from_sunset_min < 60

    # update weather conditions
    def update_conditions(self):
        self.local_weather()
        return

    # wait here until after midnight and then wait for sunrise
    # once midnight is reached + 1 second reset sunrise and sunset with correct date for new day
    def wait_until_midnight(self):
        if datetime.now() > self.sunset:
            waittime = (datetime.combine(datetime.now().date() + timedelta(days=1),
                                         datetime.strptime("0000", "%H%M").time()) - datetime.now()).total_seconds()
            waittime = waittime + 1 if waittime >= 0 else 1  # add a second and check for negative numbers
            print(f'taking a {waittime} second nap until after midnight')
            time.sleep(waittime + 60)  # wait until after midnight with a small pad just to be sure
            self.local_weather()  # reset dates and times for sunrise and sunset for a new day
        return

    # wait here until the sun is up before initialize the camera
    def wait_until_sunrise(self):
        if datetime.now() < self.sunrise:
            waittime = (self.sunrise - datetime.now()).total_seconds()
            waittime = waittime + 1 if waittime >= 0 else 1  # add a second and check for negative numbers
            print(f'taking a {waittime} second nap to wait for sun rise')
            time.sleep(waittime)  # wait until the sun comes up
        return


def main():
    spweather = CityWeather()
    spweather.update_conditions()
    print(spweather.isclear)
    print(datetime.now())
    print(spweather.sunrise)
    print(spweather.sunrise.strftime('%H:%M:%S'))
    print(spweather.sunset.strftime('%H:%M:%S'))
    print(spweather.temp)
    print(spweather.weatherdescription)
    print(spweather.windspeed)
    print(spweather.pressure)
    print(spweather.humidity)
    print(spweather.visibility)
    print(spweather.is_daytime())
    waittime = (spweather.sunrise - datetime.now()).total_seconds()
    print('waittime:', waittime)
    spweather.wait_until_midnight()
    spweather.wait_until_sunrise()
    print(spweather.is_twilight())


# test function
if __name__ == '__main__':
    main()
