import pandas as pd

def get_user_input():
    Planet_Mass = float(input("Enter the planet's mass: "))
    Planet_Radius = float(input("Enter the planet's radius: "))
    Planet_Period = float(input("Enter the planet's period: "))
    Planet_Semimajor_Axis = float(input("Enter the planet's semimajor axis: "))
    Planet_Gravity = float(input("Enter the planet's gravity: "))
    Planet_Density = float(input("Enter the planet's density: "))
    Planet_Hill_Sphere = float(input("Enter the planet's hill sphere: "))
    Planet_Distance = float(input("Enter the planet's distance: "))
    Planet_Periastron = float(input("Enter the planet's periastron: "))
    Planet_Apastron = float(input("Enter the planet's apastron: "))
    Planet_Flux = float(input("Enter the planet's flux: "))
    Planet_Equilibrium_Temperature = float(input("Enter the planet's equilibrium temperature: "))
    Planet_Surface_Temperature = float(input("Enter the planet's sruface temperature: "))
    Planet_Type = input("Enter the planet's type: ")
    Planet_Habitability_Zone_Location = float(input("Enter whether the planet is in the habitability zone: "))
    Planet_Temperature_Type = input("Enter the planet's temperature type: ")
    Planet_Habitability = None
    Planet_Earth_Similarity_Index = float(input("Enter the planet's ESI: "))
    Star_Name = None
    Star_Magnitude = float(input("Enter the host star's magnitude: "))
    Star_Distance = float(input("Enter the host star's distance: "))
    Star_Temperature = float(input("Enter the host star's temperature: "))
    Star_Mass = float(input("Enter the host star's mass: "))
    Star_Radius = float(input("Enter the host star's radius: "))
    Star_Temperature_Type = input("Enter the host star's temperature type: ")
    Star_Luminosity = float(input("Enter the host star's luminosity: "))
    Star_Snow_Line = float(input("Enter the host star's snow line: "))

    columns = [
    "Planet_Mass", "Planet_Radius", "Planet_Period", "Planet_Semimajor_Axis", "Planet_Gravity", "Planet_Density",
    "Planet_Hill_Sphere", "Planet_Distance", "Planet_Periastron", "Planet_Apastron", "Planet_Flux", 
    "Planet_Equilibrium_Temperature","Planet_Surface_Temperature", "Planet_Type", "Planet_Habitability_Zone_Location",
    "Planet_Temperature_Type", "Planet_Habitability", "Planet_Earth_Similarity_Index", "Star_Name", "Star_Magnitude",
    "Star_Distance", "Star_Temperature", "Star_Mass", "Star_Radius", "Star_Temperature_Type", "Star_Luminosity",
    "Star_Snow_Line"
    ]

    values = [
    Planet_Mass, Planet_Radius, Planet_Period, Planet_Semimajor_Axis, Planet_Gravity, Planet_Density,
    Planet_Hill_Sphere, Planet_Distance, Planet_Periastron, Planet_Apastron, Planet_Flux,
    Planet_Equilibrium_Temperature, Planet_Surface_Temperature, Planet_Type, Planet_Habitability_Zone_Location,
    Planet_Temperature_Type, Planet_Habitability, Planet_Earth_Similarity_Index, Star_Name, Star_Magnitude,
    Star_Distance, Star_Temperature, Star_Mass, Star_Radius, Star_Temperature_Type, Star_Luminosity, Star_Snow_Line
    ]

    df = pd.DataFrame([values], columns=columns)
    return df