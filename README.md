# Bird Feeder Species Classifier
This app runs on a rasp pi 4 using the Bookwork OS. The app uses picamera and tensorflow lite to process a image stream, 
detect a bird, and attempt to label the species. The object detection model is a standard google object detection model. The app is currently running a species model built by google as an example.  
That model supports 965 species. I've added some thresholds by species to skip species not present in a geo.  

## Equipment needed:
1. Rasp Pi 4 with bookworm (latest OS).
2. Camera: The code will run on just about any camera.  I tried the cheap version 1 arducam, but that generates bad pictures (5 megapixels).  bad pictures = bad predictions. I'm running a fixed focus lens right now.  Here is a link to the base; however, I would recommend trying the new autofocus  You could arducam V2.  Autofocus would be preferable and the megapixel capabilities are closer to the fixed focus I'm running (8 vs. 12 megapixels)  
   https://www.amazon.com/gp/product/B08B1QLGHS/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1  
   https://www.amazon.com/gp/product/B088GWZPL1/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1  
4. Waterproof case if you're going to run it outside.  I'd recommend this.  I drilled a hole in the bottom for the power cable and added a fan for the summer months.  The code does a temp check on the core and it should shut down if it gets too hot.  If you aren't running it outside you can tape the camera to a window.  https://www.amazon.com/gp/product/B0009NZ4KE/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&th=1&psc=1  

## Software Set up:
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
    chmod +x bash_cmds/birdclass.sh
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


## Repos
The complete project spans several 
1. Web site: https://github.com/JimMaastricht5/tweetersp  
2. Cloud storage: archive jpgs for analysis, otherwise gcs purges after 3-5 days.  Runs nightly, scheduled  https://github.com/JimMaastricht5/cloud_move_jpeg_buckets  
3. Data Aggregation: web site uses a pre-aggregation by day calculation.  That is done in this cloud function scheduled nightly: https://github.com/JimMaastricht5/cloud_data_aggregation

## Command Line Arguments
    "-cf", "--config_file", type=str, help='Config file'
    ### camera settings
    "-fc", "--flipcamera", type=bool, default=False, help="flip camera image"
    "-sw", "--screenwidth", type=int, default=640, help="max screen width"
    "-sh", "--screenheight", type=int, default=480, help="max screen height"

    ### general app settings
    "-gf", "--minanimatedframes", type=int, default=10, help="minimum number of frames with a bird"
    "-bb", "--broadcast", type=bool, default=False, help="stream images and text"
    "-v", "--verbose", type=bool, default=True, help="To tweet extra stuff or not"
    "-td", "--tweetdelay", type=int, default=1800, help="Wait time between tweets is N species seen * delay/10 with not to exceed max of tweet delay"

    ### motion and image processing settings,
    note adjustments are used as both a detector second prediction and a final
    adjustment to the output images.  # 1 no chg,< 1 -, > 1 +
    "-b", "--brightness_chg", type=int, default=1.2, help="brightness boost twilight"
    "-c", "--contrast_chg", type=float, default=1.0, help="contrast boost")  # 1 no chg,< 1 -, > 1 +
    "-cl", "--color_chg", type=float, default=1.0, help="color boost")  # 1 no chg,< 1 -, > 1 +
    "-sp", "--sharpness_chg", type=float, default=1.0, help="sharpness")  # 1 no chg,< 1 -, > 1 +

    ### prediction defaults
    "-sc", "--species_confidence", type=float, default=.90, help="species confidence threshold"
    "-bc", "--bird_confidence", type=float, default=.6, help="bird confidence threshold"
    "-ma", "--minentropy", type=float, default=5.0, help="min change from first img to current to trigger motion"
    "-ms", "--minimgperc", type=float, default=10.0, help="ignore objects that are less than % of img"
    "-hd", "--homedir", type=str, default='/home/pi/birdclassifier/', help="home directory for files"
    "-la", "--labels", type=str, default='coral.ai.inat_bird_labels.txt', help="file for species labels "
    "-tr", "--thresholds", type=str, default='coral.ai.inat_bird_threshold.csv', help="file for species thresholds"
    "-cm", "--classifier", type=str, default='coral.ai.mobilenet_v2_1.0_224_inat_bird_quant.tflite', help="model name for species classifier"

    ### feeder defaults
    "-ct", "--city", type=str, default='Madison,WI,USA', help="city name weather station uses OWM web service."
    '-fi', "--feeder_id", type=str, default=hex(uuid.getnode()), help='feeder id default MAC address'
    '-t', "--feeder_max_temp_c", type=int, default=86, help="Max operating temp for the feeder in C" 

## Arguments as a configuration file 

