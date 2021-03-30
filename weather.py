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
from datetime import datetime, timezone


# pass in city name or default to Madison, WI
# returns boolean true if the sky is clear
# sunrise and sunset as local time and full json string
def local_weather(city='Madison,WI,USA'):
    base_url = 'http://api.openweathermap.org/data/2.5/weather?q='
    full_url = base_url + city + '&appid=' + weather_key
    response = requests.get(full_url)
    # print(response.json())  # print full weather report
    # grab sunrise and sundown epoch data, parse epoch and convert to date time
    sun = str(response.json()['sys'])  # find sun rise and sunset string
    weather = str(response.json()['weather'])  # find general weather string

    try:
        start = sun.find('sunrise') + 10
        sunrise = datetime.fromtimestamp(int(sun[start: start + 10]))
        start = sun.find('sunset') + 9
        sunset = datetime.fromtimestamp(int(sun[start: start + 10]))
    except:
        sunrise = datetime.now()
        sunset = datetime.now()

    # determine cloud conditions and return true if clear
    start = weather.find('main') + 8
    skycondition = weather[start: start + 5]
    if skycondition == 'Clear': isclear = True
    else: isclear = False

    return isclear, sunrise, sunset, response.json()


def main():
    isclear, sunrise, sunset, json = local_weather()
    print(isclear)
    print(sunrise.strftime('%H:%M:%S'))
    print(sunset.strftime('%H:%M:%S'))
    print(json)


# test function
if __name__ == '__main__':
    main()
