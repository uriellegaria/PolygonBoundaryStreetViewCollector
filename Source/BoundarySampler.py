import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt

class BoundarySampler:

    def __init__(self, polygonGeometry):
        self.polygonGeometry = polygonGeometry
    

    def getMeterPoints(self):
        """
        Converts the polygon to a projected CRS (UTM) to ensure that distance calculations are in meters.
        Returns the list of boundary points in the projected CRS.
        """
        gdf = gpd.GeoDataFrame(geometry=[self.polygonGeometry], crs="EPSG:4326") 
        gdf = gdf.to_crs(epsg=32614) 
        points = list(gdf["geometry"][0].exterior.coords)
        return points


    def samplePointsInBoundary(self, deltaPhoto = 2):
        """
        Samples points along the polygon boundary at regular intervals of `deltaPhoto` meters.
        Returns a list of sampled points.
        """
        
        points = self.getMeterPoints()
        nPoints = len(points)
        distances = np.zeros(nPoints)
        sampledPoints = []

        for i in range(nPoints):  

            index1 = i%nPoints
            index2 = (i+1)%nPoints

            point1 = points[index1]
            point2 = points[index2]

            distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

            distances[index1] = distance  
        
        residual = 0
        for i in range(len(distances)):
            distance = distances[i]

            index1 = i%nPoints
            index2 = (i+1)%nPoints
            if(distance < deltaPhoto - residual):
                midPoint = (np.array(points[index1])+np.array(points[index2]))/2
                sampledPoints.append(Point(midPoint))
                residualVector = np.array(points[index2]) - midPoint
                residual = np.sqrt(residualVector[0]**2 + residualVector[1]**2)
            
            else:
                point1 = np.array(points[index1])
                point2 = np.array(points[index2])

                deltaVector = point2 - point1
                deltaSegment = np.sqrt(deltaVector[0]**2 + deltaVector[1]**2)

                initialSegmentPoint = point1 +((deltaPhoto - residual)/deltaSegment)*(deltaVector)
                sampledPoints.append(Point(initialSegmentPoint))

                deltaVectorReduced = point2 - initialSegmentPoint
                deltaVectorReducedNorm = np.sqrt(deltaVectorReduced[0]**2 + deltaVectorReduced[1]**2)
                nDivisions = int(deltaVectorReducedNorm // deltaPhoto)
                residual = deltaVectorReducedNorm - nDivisions*deltaPhoto

                for j in range(1, nDivisions+1):
                    sampledPoint = initialSegmentPoint + ((deltaPhoto * j)/deltaVectorReducedNorm)*deltaVectorReduced
                    sampledPoints.append(Point(sampledPoint))

    
        # Convert each sampled point **individually** back to EPSG:4326
        gdf = gpd.GeoDataFrame(geometry=sampledPoints, crs="EPSG:32614").to_crs(epsg=4326)

        return [(p.x, p.y) for p in gdf.geometry]

    
    def plotSampledPoints(self, deltaPhoto=2, width = 8, height = 8):

        sampledPoints = self.samplePointsInBoundary(deltaPhoto)

        # Extract coordinates directly (since both points and polygon are in EPSG:4326)
        sampled_x, sampled_y = zip(*sampledPoints)

        # Extract polygon boundary coordinates
        polygon_boundary = np.array(self.polygonGeometry.exterior.coords)
        poly_x, poly_y = polygon_boundary[:, 0], polygon_boundary[:, 1]

        # Plot
        plt.figure(figsize=(width, height))
        plt.plot(poly_x, poly_y, color="#3F76E5", linewidth=2, label="Polygon Boundary")  # Polygon in Jinx Blue
        plt.scatter(sampled_x, sampled_y, color="#FF77A8", s=20, label="Sampled Points", zorder=3)  # Points in Jinx Pink
    
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Sampled Points Along Polygon Boundary")
        plt.legend()
        plt.grid(False)
        plt.axis("off")
        plt.show()



        