import google_streetview.api
import time
import numpy as np
import math
import shutil
import os

class CollectionGeometry:
    '''
    The street geometry will collect images from both sides of the street from a start point to an end point. 
    The point geometry will collect images 360° around a point. 
    
    You can then stitch the images for analysis if needed.
    '''
    STREET = 0
    POINT = 1
    STREET_CONTINUOUS = 2
    COMBINED_VIEWS = 3
    SINGLE_SHOT = 4
    


class GoogleStreetViewCollector:
    
    
    def __init__(self, apiKey):
        self.apiKey = apiKey
        self.imagesPoint = 8
        self.size = "640x640"
        self.pitch = 20
        self.fov = 90
        
    
    def setPitch(self, pitch):
        self.pitch = pitch

    def setFOV(self, fov):
        self.fov = fov
    
    def setSize(self, width, height):
        self.size = str(width)+"x"+str(height)
    
    def setImagesStreet(self, nImagesStreet):
        self.imagesStreet = nImagesStreet
    
    def setImagesPoint(self, nImagesPoint):
        self.imagesPoint = nImagesPoint
    
    def getLocationString(self, latitude, longitude):
        return str(latitude)+","+str(longitude)

    def obtainDirectionBetweenLocations(self, location1, location2):


        lat1 = math.radians(location1[0])
        lat2 = math.radians(location2[0])

        diffLong = math.radians(location2[1] - location1[1])

        x = math.sin(diffLong) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)* math.cos(lat2)*math.cos(diffLong))

        initialBearing = math.atan2(x, y)


        initialBearing = math.degrees(initialBearing)
        compassBearing = (initialBearing + 360) % 360

        return compassBearing
        
    
    def collectPictures(self, collectionGeometry, outputPath, *args):
        if(collectionGeometry == CollectionGeometry.STREET):
            locations = args[0]
            paramsLeft = []
            paramsRight = []
            
            for i in range(0,len(locations)-1):
                if(len(locations) > 1 and i < len(locations)-1):
                    location = locations[i]
                    initialAngle = self.obtainDirectionBetweenLocations(location, locations[i+1])
                
                    paramsLeft.append({'size':self.size, 'location':self.getLocationString(location[0], location[1]), 'pitch':str(self.pitch), 'heading':str(initialAngle +270), 'key':self.apiKey,'fov':self.fov})
                    paramsRight.append({'size':self.size, 'location':self.getLocationString(location[0], location[1]), 'pitch':str(self.pitch), 'heading':str(initialAngle + 90), 'key':self.apiKey,'fov':self.fov})
                    
            
            resultsLeft = google_streetview.api.results(paramsLeft)
            pathLeft = outputPath + "/Image_street_left"
            resultsLeft.download_links(pathLeft)
            resultsRight = google_streetview.api.results(paramsRight)
            pathRight = outputPath + "/Image_street_right"
            resultsRight.download_links(pathRight)
        
        elif(collectionGeometry == CollectionGeometry.POINT):
            location = args[0]
            currentAngle = 0
            deltaHeading = 360.0/self.imagesPoint
            params = []
            
            for i in range(0,self.imagesPoint):
                
                params.append({'size':self.size, 'location':self.getLocationString(location[0], location[1]), 'pitch':str(self.pitch), 'heading':str(currentAngle), 'key':self.apiKey, 'fov':self.fov})
                currentAngle = currentAngle + deltaHeading
            results = google_streetview.api.results(params)
            path = outputPath + "/image_point"
            results.download_links(path)

        elif(collectionGeometry == CollectionGeometry.STREET_CONTINUOUS):
            locations = args[0]
            paramsLeft = []
            paramsRight = []
            
            for i in range(0,len(locations)-1):
                if(len(locations) > 1 and i < len(locations)-1):
                    location = locations[i]
                    initialAngle = self.obtainDirectionBetweenLocations(location, locations[i+1])

                    #3 angles per image
                    paramsLeft.append({'size':self.size, 'location':self.getLocationString(location[0], location[1]), 'pitch':str(self.pitch), 'heading':str(initialAngle +270-45), 'key':self.apiKey,'fov':self.fov})
                    paramsLeft.append({'size':self.size, 'location':self.getLocationString(location[0], location[1]), 'pitch':str(self.pitch), 'heading':str(initialAngle +270), 'key':self.apiKey,'fov':self.fov})
                    paramsLeft.append({'size':self.size, 'location':self.getLocationString(location[0], location[1]), 'pitch':str(self.pitch), 'heading':str(initialAngle +270+45), 'key':self.apiKey,'fov':self.fov})


                    paramsRight.append({'size':self.size, 'location':self.getLocationString(location[0], location[1]), 'pitch':str(self.pitch), 'heading':str(initialAngle + 90-45), 'key':self.apiKey,'fov':self.fov})
                    paramsRight.append({'size':self.size, 'location':self.getLocationString(location[0], location[1]), 'pitch':str(self.pitch), 'heading':str(initialAngle + 90), 'key':self.apiKey,'fov':self.fov})
                    paramsRight.append({'size':self.size, 'location':self.getLocationString(location[0], location[1]), 'pitch':str(self.pitch), 'heading':str(initialAngle + 90+45), 'key':self.apiKey,'fov':self.fov})

            
            resultsLeft = google_streetview.api.results(paramsLeft)
            pathLeft = outputPath + "/Image_street_left"
            resultsLeft.download_links(pathLeft)
            resultsRight = google_streetview.api.results(paramsRight)
            pathRight = outputPath + "/Image_street_right"
            resultsRight.download_links(pathRight)
        
        elif(collectionGeometry == CollectionGeometry.COMBINED_VIEWS):
            location = args[0]
            bearing = args[1]
            outputPathStreet = args[2]
            pointIndex = args[3]
            currentAngle = 0
            deltaHeading = 360.0/self.imagesPoint
            params = []
            minAngle = float('inf')
            bearingIndex = -1
            for i in range(0,self.imagesPoint):
                params.append({'size':self.size, 'location':self.getLocationString(location[0], location[1]), 'pitch':str(self.pitch), 'heading':str(currentAngle), 'key':self.apiKey, 'fov':self.fov})
                if(np.abs(currentAngle - bearing) < minAngle):
                    minAngle = np.abs(currentAngle - bearing)
                    bearingIndex = i
                currentAngle = currentAngle + deltaHeading
            
            leftImageIndex = (bearingIndex +int(self.imagesPoint/4))%self.imagesPoint
            rightImageIndex = (bearingIndex -int(self.imagesPoint/4))%self.imagesPoint
            results = google_streetview.api.results(params)
            pathPoint = outputPath + "/image_point"
            results.download_links(pathPoint)
            pathLeft = outputPathStreet + "/Image_street_left"
            pathRight = outputPathStreet + "/Image_street_right"

            if(not os.path.exists(pathLeft)):
                os.makedirs(pathLeft)
            
            if(not os.path.exists(pathRight)):
                os.makedirs(pathRight)

            #Copy the left-side image file 
            fromPathLeft = pathPoint + "/gsv_"+str(leftImageIndex)+".jpg"
            toPathLeft = pathLeft +"/gsv_"+str(pointIndex)+".jpg"
            if(os.path.exists(fromPathLeft)):
                shutil.copyfile(fromPathLeft, toPathLeft)
            
            #Copy the right-side image file
            fromPathRight = pathPoint + "/gsv_"+str(rightImageIndex)+".jpg"
            toPathRight = pathRight +"/gsv_"+str(pointIndex)+".jpg"
            if(os.path.exists(fromPathRight) and os.path.exists(toPathLeft)):
                shutil.copyfile(fromPathRight, toPathRight)
        
        elif(collectionGeometry == CollectionGeometry.SINGLE_SHOT):
            location = args[0]
            headingAngle = args[1]

            singlePictureParams = {'size':self.size, 'location':self.getLocationString(location[0], location[1]), 'pitch':str(self.pitch), 'heading':str(headingAngle), 'key':self.apiKey, 'fov':self.fov}
            results = google_streetview.api.results([singlePictureParams])
            imageFolderName = str(int(headingAngle))
            completePath = os.path.join(outputPath, imageFolderName)

            if(not os.path.exists(completePath)):
                os.makedirs(completePath)

            results.download_links(completePath)

            return os.path.join(completePath, "gsv_0.jpg")
            




        
            






            