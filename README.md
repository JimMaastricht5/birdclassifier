# bird classifier
libraries for face detection and tracking as well as bird object detection and species classification using a rasp pi

birdclass uses picamera and tensorflow to process a video stream, detect a bird, and attempt to label the species.  Object detection model is a standard google object detection model.  I've tried customer developing species models using a variety of techniques. Accuracy of ~98% during training.  Less successful in the wild when it is gray and gloomy outside.  I am currently running a species model built by google as an example.  That model supports 965 species. I've added some thresholds by species to elminate species not present in a geo.  

Equipment needed:
1. Rasp Pi 4 with OS.
2. Camera: The code will run on just about any camera.  I tried the cheap version 1 arducam, but that generates bad pictures (5 megapixels).  bad pictures = bad predictions. I'm running a fixed focus lens right now.  Here is a link to the base; however, I would recommend trying the new autofocus  You could arducam V2.  Autofocus would be preferable and the megapixel capabilities are closer to the fixed focus I'm running (8 vs. 12 megapixels)
3. Waterproof case if you're going to run it outside.  I'd recommend this.  I drilled a hole in the bottom for the power cable and added a fan for the summer months.  The code does a temp check on the core and it should shut down if it gets too hot.  If you aren't running it outside you can tape the camera to a window 
Software:
0. Setup VNC connect on the Pi so you can work with it remotely
1. Install and enable the camera
2. Install Python 
3. Setup a developer Twitter account for the bot to broadcast to...  twitter has good directions
4. Setup a free account on OpenWeatherMap (http://api.openweathermap.org)
5. Clone the bird classifier project: https://github.com/JimMaastricht5/birdclassifier
6. create your auth.py file with the Twitter keys and the open weather key.  When you get to this point let me know and we can work together on it.  Twitter: api_key, api_secret_key, access_token, access_token_secretOMW: weather_key
7. I have some bash scripts to start the process and can share those too when you get here.  I just haven't gotten around to putting them in git.  
8. tweek the parameters for your use case.  I have the threshold set really high so I get less confusing results, but that might not trigger enough for you.  
