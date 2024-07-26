# bird classifier
birdclass uses picamera and tensorflow lite to process a image stream, detect a bird, and attempt to label the species.  Object detection model is a standard google object detection model.  I am currently running a species model built by google as an example.  That model supports 965 species. I've added some thresholds by species to elminate species not present in a geo.  

Equipment needed:
1. Rasp Pi 4 with bookworm (latest OS).
2. Camera: The code will run on just about any camera.  I tried the cheap version 1 arducam, but that generates bad pictures (5 megapixels).  bad pictures = bad predictions. I'm running a fixed focus lens right now.  Here is a link to the base; however, I would recommend trying the new autofocus  You could arducam V2.  Autofocus would be preferable and the megapixel capabilities are closer to the fixed focus I'm running (8 vs. 12 megapixels)  
   https://www.amazon.com/gp/product/B08B1QLGHS/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1  
   https://www.amazon.com/gp/product/B088GWZPL1/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1  
4. Waterproof case if you're going to run it outside.  I'd recommend this.  I drilled a hole in the bottom for the power cable and added a fan for the summer months.  The code does a temp check on the core and it should shut down if it gets too hot.  If you aren't running it outside you can tape the camera to a window.  https://www.amazon.com/gp/product/B0009NZ4KE/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&th=1&psc=1  

Software:
1. Use the Rasp Pi Imager to create the OS on a mini-sd card  
2. Boot the system and update the software  
     sudo apt upgrade  
     sudo apt update  
3.  Check that the locale, and time are correct in the Pi/Perferences/Raspberry Pi Configuration menu  
4.  The bookworm version of the OS enables the camera by default.  
5. Python should be installed already.  Check to see if it is 3.11 or higher.  
   python -V
6. Otherwise install Python with the following commands:
   sudo add-apt-repository ppa:deadsnakes/ppa  
   sudo apt update  
   sudo apt install python3.11 
   sudo apt install python3-pip  
   python3.11 --version  
7. Picamera2 should also be installed.  Verify that with pip3 and install it if necessary with the next command  
   pip3 show picamera2  
   sudo apt install python3-picamera2  
8. Install other software  
   (remote control software from windows with RDP): sudo apt install xrdp  
10. Setup the software and the virtual environment for python, note picamera2 is hard to install and is installed globally so we'll use that instead.  We will also run a camera test that writes out a file to the assets directory
    mkdir birdclassifiercd 
    git clone https://github.com/JimMaastricht5/birdclassifier.git
    cd birdclassifier
    ls
    mkdir assets
    python -m venv venv --system-site-packages
    pip install --upgrade pip
    pip install -r requirements.txt
    bash /home/pi/birdclassifier/bash_cmds/cameratest2.sh
9. [Note: add link here] Setup a developer Twitter account for the bot to broadcast to...  twitter has good directions  
10. Setup a free account on OpenWeatherMap (http://api.openweathermap.org)cat   
11. create your auth.py file with the Twitter keys and the open weather key.  
    set a file in the birdclassifier directory and call it auth.py. I would suggest using "nano auth.py"  
    enter the keys from your twitter account into the file. 
   \# Twitter keys  
   api_key =''   
   api_secret_key =''   
   access_token= ''  
   access_token_secret = ''   
   bear_token = ''  
   client_id = ''  
   client_secret = ''  
   \# weather  
   weather_key = ''  
   \# google  
   google_json_key = ''  
13. Note: what about google GCS containesd in the json_key_file?  
14. [Note: add crontab setup] I have some bash scripts to start the process every day on a schedule.  


The complete project spans several repos:  
1. Web site: https://github.com/JimMaastricht5/tweetersp  
2. Cloud storage: archive jpgs for analysis, otherwise gcs purges after 3-5 days.  Runs nightly, scheduled  https://github.com/JimMaastricht5/cloud_move_jpeg_buckets  
3. Data Aggregation: web site uses a pre-aggregation by day calculation.  That is done in this cloud function scheduled nightly: https://github.com/JimMaastricht5/cloud_data_aggregation  
