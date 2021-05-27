# bird classifier
libraries for face detection and tracking as well as bird object detection and species classification using a rasp pi

birdclass uses opencv and tensor to process a video stream, detect a bird, and attempt to label the species.  Object detection model is a standard google object detection model.  I've tried customer developed species models using a variety of techniques. Accuracy of ~98% during training.  Less successful in the wild when it is gray and gloomy outside.  I am currently running a species model built by google as an example.  That model supports 965 species. I've added some thresholds by species to elminate species not present in a geo.  

pyface2 performs face detection and object tracking using a pantilt mechanism.  
Buy one on amazon here: https://www.amazon.com/gp/product/B07PQ39C6B/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1
