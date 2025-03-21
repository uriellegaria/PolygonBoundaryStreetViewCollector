from Source.Data_Retrieval.StreetViewTools import GoogleStreetViewCollector
from Source.Data_Retrieval.StreetViewTools import CollectionGeometry
from datetime import datetime
from shapely.geometry import Point, mapping
import os
import json

class ParkPhotographerGeoJSON:
    
    def __init__(self, apiKey):
        self.collector = GoogleStreetViewCollector(apiKey)
        self.parksData = [] 
        self.parkPolygonStored = False  

    def adjustHeadingForTrueNorth(self, headingAngle):
        """
        Converts from Cartesian standard angle to Google Street View heading.
        """
        adjustedHeading = (90 - headingAngle) % 360
        return adjustedHeading

    def calculateAngleQuality(self, adjustedHeading):
        """
        Computes angle quality:
        - **1.0** = Perfect (90 degrees)
        - **0.0** = Worst (0, 180, 360 degrees)
        - **Handles circular angle corrections**
        """
        deviation = min(abs(90 - adjustedHeading), abs(270 - adjustedHeading))
        return max(0, 1 - (deviation / 90))

    def captureParkImage(self, optimalPoint, outputFolder):
        """
        Captures a single image from an observation point.
        The image will be stored inside `outputFolder` by Street View API.
        """
        latitude = optimalPoint['latitude']
        longitude = optimalPoint['longitude']
        headingAngle = optimalPoint['heading']

        adjustedHeading = self.adjustHeadingForTrueNorth(headingAngle)

        return self.collector.collectPictures(
            CollectionGeometry.SINGLE_SHOT, 
            outputFolder,  
            [latitude, longitude], 
            adjustedHeading
        )
    
    def captureMultipleViewsForPark(self, optimalPointsList, outputBaseFolder, parkPolygon, verbose=True):
        """
        Captures images for multiple observation points and stores their metadata.
        Each image is stored in its respective folder inside `outputBaseFolder`.
        """
        for idx, optimalPoint in enumerate(optimalPointsList):
            targetPoint = Point(optimalPoint["longitude"], optimalPoint["latitude"])
            viewFolder = os.path.join(outputBaseFolder, f"park_view_{idx}")  
            
            self.captureAndRegisterImage(
                optimalPoint, viewFolder, parkPolygon, targetPoint, outputBaseFolder, verbose=verbose
            )

    def captureAndRegisterImage(self, optimalPoint, outputFolder, parkPolygon, targetPoint, outputBaseFolder, captureConditions="", verbose=True):
        """
        Captures an image of the park and stores its metadata in a GeoJSON structure.
        The `imagePath` stored in GeoJSON is relative to `outputBaseFolder`.
        """
        os.makedirs(outputFolder, exist_ok=True)

        latitude = optimalPoint["latitude"]
        longitude = optimalPoint["longitude"]
        headingAngle = optimalPoint["heading"]

        adjustedHeading = self.adjustHeadingForTrueNorth(headingAngle)
        angleQuality = self.calculateAngleQuality(90 - optimalPoint["orthogonality_deviation"])

        pathImage = self.collector.collectPictures(
            CollectionGeometry.SINGLE_SHOT, 
            outputFolder,  
            [latitude, longitude], 
            adjustedHeading
        )

        success = os.path.exists(pathImage)

        if success:
            relativeImagePath = os.path.relpath(pathImage, start=os.path.dirname(outputBaseFolder))

            currentDate = datetime.today().date().isoformat()  
            observerLocation = Point(longitude, latitude)  
            distanceToTarget = optimalPoint["distance"]

            parkFeature = {
                "type": "Feature",
                "geometry": mapping(targetPoint),
                "properties": {
                    "imagePath": relativeImagePath,  
                    "captureDate": currentDate,
                    "observerLocation": mapping(observerLocation),  
                    "headingAngle": adjustedHeading,
                    "distanceToTarget": distanceToTarget,
                    "captureConditions": captureConditions,
                    "angleQuality": angleQuality,  
                }
            }

            if not self.parkPolygonStored:
                parkFeature["parkGeometry"] = mapping(parkPolygon)
                self.parkPolygonStored = True 

            self.parksData.append(parkFeature)
            if verbose:
                print(f"Image captured and registered: {relativeImagePath}")
        else:
            if verbose:
                print("Image capture failed.")

    def exportGeoJSON(self, outputPath):
        """
        Exports the stored park observation data to a GeoJSON file.
        """
        geoJsonData = {
            "type": "FeatureCollection",
            "features": self.parksData
        }

        with open(outputPath, 'w') as f:
            json.dump(geoJsonData, f, indent=4)
        print(f"GeoJSON data exported to {outputPath}.")
