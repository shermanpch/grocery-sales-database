import pandas as pd

# Reference date for the dataset
REFERENCE_DATE = pd.Timestamp("2018-05-10")

# Earth's radius in kilometers
EARTH_RADIUS_KM = 6371.0

# Reference coordinate (New York City)
REFERENCE_LAT = 40.7128
REFERENCE_LON = -74.0060

# Coordinates for each city (latitude, longitude)
city_coordinates = {
    "New York": (40.7128, -74.0060),
    "Los Angeles": (34.0522, -118.2437),
    "Chicago": (41.8781, -87.6298),
    "Houston": (29.7604, -95.3698),
    "Phoenix": (33.4484, -112.0740),
    "Philadelphia": (39.9526, -75.1652),
    "San Antonio": (29.4241, -98.4936),
    "San Diego": (32.7157, -117.1611),
    "Dallas": (32.7767, -96.7970),
    "San Jose": (37.3382, -121.8863),
    "Austin": (30.2672, -97.7431),
    "Jacksonville": (30.3322, -81.6557),
    "Fort Worth": (32.7555, -97.3308),
    "Columbus": (39.9612, -82.9988),
    "Charlotte": (35.2271, -80.8431),
    "San Francisco": (37.7749, -122.4194),
    "Indianapolis": (39.7684, -86.1581),
    "Seattle": (47.6062, -122.3321),
    "Denver": (39.7392, -104.9903),
    "Washington": (38.9072, -77.0369),
    "Boston": (42.3601, -71.0589),
    "El Paso": (31.7619, -106.4850),
    "Detroit": (42.3314, -83.0458),
    "Nashville": (36.1627, -86.7816),
    "Memphis": (35.1495, -90.0490),
    "Portland": (45.5152, -122.6784),
    "Oklahoma": (35.4676, -97.5164),
    "Las Vegas": (36.1699, -115.1398),
    "Louisville": (38.2527, -85.7585),
    "Baltimore": (39.2904, -76.6122),
    "Milwaukee": (43.0389, -87.9065),
    "Albuquerque": (35.0844, -106.6504),
    "Tucson": (32.2226, -110.9747),
    "Fresno": (36.7378, -119.7871),
    "Sacramento": (38.5816, -121.4944),
    "Mesa": (33.4152, -111.8315),
    "Kansas": (39.1142, -94.6275),
    "Atlanta": (33.7490, -84.3880),
    "Colorado": (39.5501, -105.7821),
    "Miami": (25.7617, -80.1918),
    "Tulsa": (36.1540, -95.9928),
    "Oakland": (37.8044, -122.2712),
    "Minneapolis": (44.9778, -93.2650),
    "Arlington": (32.7357, -97.1081),
    "Newark": (40.7357, -74.1724),
    "Anchorage": (61.2181, -149.9003),
    "Honolulu": (21.3069, -157.8583),
    "Wichita": (37.6872, -97.3301),
    "Cleveland": (41.4993, -81.6944),
    "Tampa": (27.9506, -82.4572),
    "Bakersfield": (35.3733, -119.0187),
    "Aurora": (39.7294, -104.8319),
    "St. Louis": (38.6270, -90.1994),
    "Raleigh": (35.7796, -78.6382),
    "Pittsburgh": (40.4406, -79.9959),
    "St. Paul": (44.9537, -93.0900),
    "Cincinnati": (39.1031, -84.5120),
    "Greensboro": (36.0726, -79.7920),
    "Toledo": (41.6528, -83.5379),
    "New Orleans": (29.9511, -90.0715),
    "Lincoln": (40.8136, -96.7026),
    "Buffalo": (42.8864, -78.8784),
    "Norfolk": (36.8508, -76.2859),
    "Rochester": (43.1566, -77.6088),
    "Jersey": (40.7282, -74.0776),
    "Mobile": (30.6954, -88.0399),
    "Little Rock": (34.7465, -92.2896),
    "Dayton": (39.7589, -84.1916),
    "Shreveport": (32.5252, -93.7502),
    "Madison": (43.0731, -89.4012),
    "Richmond": (37.5407, -77.4360),
    "Glendale": (33.5387, -112.1860),
    "Yonkers": (40.9312, -73.8988),
    "Hialeah": (25.8576, -80.2781),
    "Fort Wayne": (41.0793, -85.1394),
    "Garland": (32.9126, -96.6389),
    "Lubbock": (33.5779, -101.8552),
    "Fremont": (37.5483, -121.9886),
    "Santa Ana": (33.7455, -117.8677),
    "Stockton": (37.9577, -121.2908),
    "Des Moines": (41.5868, -93.6250),
    "Anaheim": (33.8353, -117.9145),
    "Birmingham": (33.5437, -86.7796),
    "Montgomery": (32.3617, -86.2792),
    "Grand Rapids": (42.9638, -85.6700),
    "Spokane": (47.6597, -117.4291),
    "Baton Rouge": (30.4712, -91.1474),
    "Jackson": (32.2989, -90.1847),
    "Akron": (41.0818, -81.5115),
    "Tacoma": (47.2458, -122.4594),
    "Virginia Beach": (36.8631, -76.0158),
    "Corpus Christi": (27.8006, -97.3964),
    "Long Beach": (33.7701, -118.1937),
    "Riverside": (33.9534, -117.3962),
    "Omaha": (41.2572, -95.9951),
    "St. Petersburg": (27.7731, -82.6400),
}
