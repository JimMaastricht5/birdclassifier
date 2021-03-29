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
# As OpenWeatherMap APIs need a valid API key to allow responses, PyOWM won't work if you don't provide one.
# This stands for both free and paid (pro) subscription plans.
# You can signup for a free API key on the OWM website
# Please notice that the free API subscription plan is subject to requests throttling.
from auth import (
    weather_key
)
import requests

# pass in city name or default to sun prairie, WI
def local_weather(city='Sun&Prairie'):
    base_url = 'http://api.openweathermap.org/data/2.5/weather?q='
    full_url= base_url + city +'&appid=' + weather_key
    response = requests.get(full_url)
    weather = str(response.json()['weather'])
    print(weather)
    start = weather.find('main')+ 8
    skycondition = weather[start: start + 5]
    if skycondition == 'Clear':
        isclear = True
    else:
        isclear = False
    return isclear


def main():
    print(local_weather())


# test function
if __name__ == '__main__':
    main()
