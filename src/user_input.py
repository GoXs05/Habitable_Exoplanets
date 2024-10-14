import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

df = None

# Function to handle input validation and data collection
def submit_data():
    global df
    try:
        # Helper function to handle empty entries for numerical values
        def parse_float(entry):
            return float(entry) if entry else None

        # Collecting inputs from the UI with handling for empty/null values
        inputs = {
            'Planet_Mass': parse_float(planet_mass_entry.get()),
            'Planet_Radius': parse_float(planet_radius_entry.get()),
            'Planet_Period': parse_float(planet_period_entry.get()),
            'Planet_Semimajor_Axis': parse_float(planet_semimajor_entry.get()),
            'Planet_Gravity': parse_float(planet_gravity_entry.get()),
            'Planet_Density': parse_float(planet_density_entry.get()),
            'Planet_Hill_Sphere': parse_float(planet_hill_entry.get()),
            'Planet_Distance': parse_float(planet_distance_entry.get()),
            'Planet_Periastron': parse_float(planet_periastron_entry.get()),
            'Planet_Apastron': parse_float(planet_apastron_entry.get()),
            'Planet_Flux': parse_float(planet_flux_entry.get()),
            'Planet_Equilibrium_Temperature': parse_float(planet_eq_temp_entry.get()),
            'Planet_Surface_Temperature': parse_float(planet_surface_temp_entry.get()),
            'Planet_Type': planet_type_var.get() if planet_type_var.get() else None,
            'Planet_Habitability_Zone_Location': parse_float(planet_habitability_zone_entry.get()),
            'Planet_Temperature_Type': planet_temp_type_var.get() if planet_temp_type_var.get() else None,
            'Planet_Habitability': None,
            'Planet_Earth_Similarity_Index': parse_float(planet_esi_entry.get()),
            'Star_Name': None,
            'Star_Magnitude': parse_float(star_magnitude_entry.get()),
            'Star_Distance': parse_float(star_distance_entry.get()),
            'Star_Temperature': parse_float(star_temp_entry.get()),
            'Star_Mass': parse_float(star_mass_entry.get()),
            'Star_Radius': parse_float(star_radius_entry.get()),
            'Star_Temperature_Type': star_temp_type_var.get() if star_temp_type_var.get() else None,
            'Star_Luminosity': parse_float(star_luminosity_entry.get()),
            'Star_Snow_Line': parse_float(star_snow_line_entry.get())
        }

        # Create columns and values for the DataFrame
        columns = [
            "Planet_Mass", "Planet_Radius", "Planet_Period", "Planet_Semimajor_Axis", "Planet_Gravity", "Planet_Density",
            "Planet_Hill_Sphere", "Planet_Distance", "Planet_Periastron", "Planet_Apastron", "Planet_Flux", 
            "Planet_Equilibrium_Temperature","Planet_Surface_Temperature", "Planet_Type", "Planet_Habitability_Zone_Location",
            "Planet_Temperature_Type", "Planet_Habitability", "Planet_Earth_Similarity_Index", "Star_Name", "Star_Magnitude",
            "Star_Distance", "Star_Temperature", "Star_Mass", "Star_Radius", "Star_Temperature_Type", "Star_Luminosity",
            "Star_Snow_Line"
        ]
        
        values = [inputs[col] for col in columns]
        
        # Update global DataFrame (use pd.NA instead of None for better handling in pandas)
        df = pd.DataFrame([values], columns=columns).replace({None: pd.NA})
        
        messagebox.showinfo("Success", "Data collected successfully!")
        root.quit()
    except ValueError as e:
        messagebox.showerror("Input Error", f"Invalid input: {e}")


# Creating the Tkinter window
root = tk.Tk()
root.title("Planet and Star Input Form")

# Create labels and input fields for planets & stars
tk.Label(root, text="Planet Mass:").grid(row=0, column=0)
planet_mass_entry = tk.Entry(root)
planet_mass_entry.grid(row=0, column=1)

tk.Label(root, text="Planet Radius:").grid(row=1, column=0)
planet_radius_entry = tk.Entry(root)
planet_radius_entry.grid(row=1, column=1)

tk.Label(root, text="Planet Period:").grid(row=2, column=0)
planet_period_entry = tk.Entry(root)
planet_period_entry.grid(row=2, column=1)

tk.Label(root, text="Planet Semimajor Axis:").grid(row=3, column=0)
planet_semimajor_entry = tk.Entry(root)
planet_semimajor_entry.grid(row=3, column=1)

tk.Label(root, text="Planet Gravity:").grid(row=4, column=0)
planet_gravity_entry = tk.Entry(root)
planet_gravity_entry.grid(row=4, column=1)

tk.Label(root, text="Planet Density:").grid(row=5, column=0)
planet_density_entry = tk.Entry(root)
planet_density_entry.grid(row=5, column=1)

tk.Label(root, text="Planet Hill Sphere:").grid(row=6, column=0)
planet_hill_entry = tk.Entry(root)
planet_hill_entry.grid(row=6, column=1)

tk.Label(root, text="Planet Distance:").grid(row=7, column=0)
planet_distance_entry = tk.Entry(root)
planet_distance_entry.grid(row=7, column=1)

tk.Label(root, text="Planet Periastron:").grid(row=8, column=0)
planet_periastron_entry = tk.Entry(root)
planet_periastron_entry.grid(row=8, column=1)

tk.Label(root, text="Planet Apastron:").grid(row=9, column=0)
planet_apastron_entry = tk.Entry(root)
planet_apastron_entry.grid(row=9, column=1)

tk.Label(root, text="Planet Flux:").grid(row=10, column=0)
planet_flux_entry = tk.Entry(root)
planet_flux_entry.grid(row=10, column=1)

tk.Label(root, text="Planet Equilibrium Temperature:").grid(row=11, column=0)
planet_eq_temp_entry = tk.Entry(root)
planet_eq_temp_entry.grid(row=11, column=1)

tk.Label(root, text="Planet Surface Temperature:").grid(row=12, column=0)
planet_surface_temp_entry = tk.Entry(root)
planet_surface_temp_entry.grid(row=12, column=1)

tk.Label(root, text="Planet Type:").grid(row=13, column=0)
planet_type_var = tk.StringVar(value="Terran")
planet_type_menu = ttk.Combobox(root, textvariable=planet_type_var, values=["Jovian", "Neptunian", "Terran", "Superterran"])
planet_type_menu.grid(row=13, column=1)

tk.Label(root, text="Planet Habitability Zone Location:").grid(row=14, column=0)
planet_habitability_zone_entry = tk.Entry(root)
planet_habitability_zone_entry.grid(row=14, column=1)

tk.Label(root, text="Planet Temperature Type:").grid(row=15, column=0)
planet_temp_type_var = tk.StringVar(value="Warm")
planet_temp_type_menu = ttk.Combobox(root, textvariable=planet_temp_type_var, values=["Hot", "Cold", "Warm"])
planet_temp_type_menu.grid(row=15, column=1)

tk.Label(root, text="Planet Earth Similarity Index:").grid(row=16, column=0)
planet_esi_entry = tk.Entry(root)
planet_esi_entry.grid(row=16, column=1)

tk.Label(root, text="Star Magnitude:").grid(row=17, column=0)
star_magnitude_entry = tk.Entry(root)
star_magnitude_entry.grid(row=17, column=1)

tk.Label(root, text="Star Distance:").grid(row=18, column=0)
star_distance_entry = tk.Entry(root)
star_distance_entry.grid(row=18, column=1)

tk.Label(root, text="Star Temperature:").grid(row=19, column=0)
star_temp_entry = tk.Entry(root)
star_temp_entry.grid(row=19, column=1)

tk.Label(root, text="Star Mass:").grid(row=20, column=0)
star_mass_entry = tk.Entry(root)
star_mass_entry.grid(row=20, column=1)

tk.Label(root, text="Star Radius:").grid(row=21, column=0)
star_radius_entry = tk.Entry(root)
star_radius_entry.grid(row=21, column=1)

tk.Label(root, text="Star Temperature Type:").grid(row=22, column=0)
star_temp_type_var = tk.StringVar(value="G")
star_temp_type_menu = ttk.Combobox(root, textvariable=star_temp_type_var, values=["K", "F", "G", "M", "A"])
star_temp_type_menu.grid(row=22, column=1)

tk.Label(root, text="Star Luminosity:").grid(row=23, column=0)
star_luminosity_entry = tk.Entry(root)
star_luminosity_entry.grid(row=23, column=1)

tk.Label(root, text="Star Snow Line:").grid(row=24, column=0)
star_snow_line_entry = tk.Entry(root)
star_snow_line_entry.grid(row=24, column=1)

# Submit button
submit_button = tk.Button(root, text="Submit", command=submit_data)
submit_button.grid(row=25, column=0, columnspan=2)

def run_UI():
    # Start the Tkinter main loop
    root.mainloop()

    print(df)
    return df
