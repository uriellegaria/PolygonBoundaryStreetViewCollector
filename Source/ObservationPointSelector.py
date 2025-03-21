import numpy as np
import geopandas as gpd
from rtree import index
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from tqdm import tqdm
import matplotlib.pyplot as plt

class ParkObservationPointSelector:
    '''
    Selects observation points based on how close they are to the boundary
    and orthogonality to the boundary, it also checks for occlusions so that you don't try to observe a face
    of the park from the face behind it. 

    I used GPT to help me with the implementation of some of the methods, like using R-tree to search for 
    intersections. I need to document these classes better. 
    '''
    def __init__(self, boundaryPointsGDF, streetNetworkGDF, parkPolygon,
                 obstacleGDF=None, projectedCRS="EPSG:6372"):
        self.projectedCRS = projectedCRS

        self.boundaryPointsGDF = boundaryPointsGDF.to_crs(self.projectedCRS)
        self.streetNetworkGDF = streetNetworkGDF.to_crs(self.projectedCRS)
        self.parkPolygon = gpd.GeoSeries([parkPolygon], crs="EPSG:4326") \
                              .to_crs(self.projectedCRS).iloc[0]
        self.obstacleGDF = None
        if obstacleGDF is not None:
            self.obstacleGDF = obstacleGDF.to_crs(self.projectedCRS)

        self.streetIdx = self._buildSpatialIndex(self.streetNetworkGDF)

    def _buildSpatialIndex(self, streetNetworkGDF):
        idx = index.Index()
        for i, geom in enumerate(streetNetworkGDF.geometry):
            idx.insert(i, geom.bounds)
        return idx

    def _localBoundaryTangent(self, boundaryPoint, offset=0.5):
        """
        Compute a small local tangent vector at boundaryPoint on the park boundary
        by sampling the exterior ring from dist-offset to dist+offset.
        """
        ring = self.parkPolygon.exterior
        dist_along_ring = ring.project(boundaryPoint)

        d0 = max(dist_along_ring - offset, 0)
        d1 = min(dist_along_ring + offset, ring.length)

        p0 = ring.interpolate(d0)
        p1 = ring.interpolate(d1)
        return np.array([p1.x - p0.x, p1.y - p0.y])

    def _angleDeviationFrom90(self, vecA, vecB):
        magA, magB = np.linalg.norm(vecA), np.linalg.norm(vecB)
        if magA < 1e-12 or magB < 1e-12:
            return np.nan
        cosTheta = np.dot(vecA, vecB) / (magA * magB)
        angle = np.degrees(np.arccos(np.clip(cosTheta, -1.0, 1.0)))
        return abs(90 - angle)

    def _calculateHeading(self, vantage, target):
        dx = target.x - vantage.x
        dy = target.y - vantage.y
        return np.degrees(np.arctan2(dy, dx))

    def checkLineOfSight(self, vantage, boundaryPoint, shrink=0.0):
        """
        True if vantage->boundary does NOT pass the park interior or obstacles.
        Optionally shrink the park polygon by 'shrink' meters so lines hugging
        the boundary are also blocked.
        """
        if self.parkPolygon.contains(vantage):
            return False

        line = LineString([vantage, boundaryPoint])

        if shrink > 0:
            parkTest = self.parkPolygon.buffer(-shrink)
        else:
            parkTest = self.parkPolygon

        line_minus_park = line.difference(parkTest)
        if not line_minus_park.equals_exact(line, tolerance=1e-9):
            return False

        if self.obstacleGDF is not None:
            for _, obs in self.obstacleGDF.iterrows():
                if line.intersects(obs.geometry):
                    return False

        return True

    def selectOptimalPoints(self, distanceThreshold=30, bufferSize=0.001,
                            orthogonalityOffset=0.5, 
                            lineOfSightShrink=0.0):
        """
        Now uses local tangent for orthogonality, and checks line-of-sight with optional 
        negative buffer to fully block near-boundary lines.
        """
        optimalPoints = []
        skipped = []

        for targetIdx, row in tqdm(self.boundaryPointsGDF.iterrows(),
                                   total=len(self.boundaryPointsGDF)):
            targetPoint = row.geometry
            localTangent = self._localBoundaryTangent(targetPoint, offset=orthogonalityOffset)

            buff = targetPoint.buffer(bufferSize).bounds
            nearbyIdx = list(self.streetIdx.intersection(buff))
            candidateStreets = self.streetNetworkGDF.iloc[nearbyIdx]

            if candidateStreets.empty:
                distSeries = self.streetNetworkGDF.geometry.distance(targetPoint)
                candidateStreets = self.streetNetworkGDF[distSeries <= distanceThreshold]

            best = None
            bestAngleDev = float('inf')
            bestDist = float('inf')

            for _, streetRow in candidateStreets.iterrows():
                streetGeom = streetRow.geometry
                if streetGeom.is_empty:
                    continue

                projDist = streetGeom.project(targetPoint)
                vantageGeom = streetGeom.interpolate(projDist)

                distToTarget = vantageGeom.distance(targetPoint)
                if distToTarget > distanceThreshold:
                    continue

                if not self.checkLineOfSight(vantageGeom, targetPoint, 
                                             shrink=lineOfSightShrink):
                    continue

                viewVec = np.array([targetPoint.x - vantageGeom.x,
                                    targetPoint.y - vantageGeom.y])
                angleDev = self._angleDeviationFrom90(localTangent, viewVec)

                if angleDev < bestAngleDev or (angleDev == bestAngleDev and distToTarget < bestDist):
                    bestAngleDev = angleDev
                    bestDist = distToTarget
                    best = {
                        "longitude": vantageGeom.x,
                        "latitude": vantageGeom.y,
                        "heading": self._calculateHeading(vantageGeom, targetPoint),
                        "distance": distToTarget,
                        "targetIdx": targetIdx,
                        "orthogonality_deviation": angleDev
                    }

            if best:
                optimalPoints.append(best)
            else:
                skipped.append(targetIdx)

        print("Skipped:", len(skipped), "boundary points")

        vantageGDF = gpd.GeoDataFrame(
            optimalPoints,
            geometry=gpd.points_from_xy(
                [p["longitude"] for p in optimalPoints],
                [p["latitude"] for p in optimalPoints]
            ),
            crs=self.projectedCRS
        ).to_crs("EPSG:4326")

        for i, geom in enumerate(vantageGDF.geometry):
            optimalPoints[i]["longitude"] = geom.x
            optimalPoints[i]["latitude"] = geom.y

        return optimalPoints, skipped

    def plotObservationPoints(self, pillow=50, width=8, height=8, pointSize=5, distanceThreshold = 30):
 
        park_gdf = gpd.GeoDataFrame(geometry=[self.parkPolygon], crs=self.projectedCRS)
        boundary_gdf = self.boundaryPointsGDF.to_crs(self.projectedCRS)
        streets_gdf = self.streetNetworkGDF.to_crs(self.projectedCRS)

        vantagePoints, _ = self.selectOptimalPoints(distanceThreshold = distanceThreshold)
        if not vantagePoints:
            print(" No observation points found.")
            return

        vantageGDF = gpd.GeoDataFrame(
            vantagePoints,
            geometry=gpd.points_from_xy(
                [p["longitude"] for p in vantagePoints],
                [p["latitude"] for p in vantagePoints]
            ),
            crs="EPSG:4326"
        ).to_crs(self.projectedCRS)

        minx, miny, maxx, maxy = park_gdf.total_bounds
        if any(np.isnan([minx, miny, maxx, maxy])):
            print("Park polygon bounds are NaN; invalid geometry?")
            return
        
        minx -= pillow
        maxx += pillow
        miny -= pillow
        maxy += pillow

        streets_gdf = streets_gdf.cx[minx:maxx, miny:maxy]

        plt.figure(figsize=(width, height))

        px, py = self.parkPolygon.exterior.xy
        plt.plot(px, py, color="blue", linewidth=2, label="Park Boundary")

        plt.scatter(
            boundary_gdf.geometry.x, boundary_gdf.geometry.y,
            color="magenta", s=pointSize, label="Sampled Points", zorder=3
        )

        streets_gdf.plot(ax=plt.gca(), color="gray", linewidth=1, label="Nearby Streets")

        plt.scatter(
            vantageGDF.geometry.x, vantageGDF.geometry.y,
            color="green", s=pointSize, label="Observation Points", zorder=4
        )

        for i, vantageRow in vantageGDF.iterrows():
            vantageGeom = vantageRow.geometry
            boundaryIdx = vantageRow["targetIdx"] 

            boundaryGeom = boundary_gdf.loc[boundaryIdx, "geometry"]

            line = LineString([vantageGeom, boundaryGeom])
            x_l, y_l = line.xy
            plt.plot(x_l, y_l, color="gray", linewidth=0.5, alpha=0.7)

        plt.title("Observation Point Selection")
        plt.legend()
        plt.axis("equal")
        plt.axis("off")
        plt.xlim(minx, maxx)
        plt.ylim(miny, maxy)
