import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QTabWidget, QFrame, QStyleFactory
)
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton
from PyQt5.QtWebEngineWidgets import QWebEngineView
import folium
import os
import datetime
from datetime import datetime, timedelta
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.interpolate import make_interp_spline
import numpy as np



# Load models
param_model = joblib.load(r"EM Project\Saved_Models\saved_model_parameters.pkl")
power_model = joblib.load(r"EM Project\Saved_Models\saved_model_pout.pkl")

class SolarParkGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Solar Park Location Finder")
        self.showMaximized()
        self.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")

        # Initialize instance variables for locations
        self.lat1 = None
        self.lon1 = None
        self.lat2 = None
        self.lon2 = None
        self.is_first_location_set = False  # Track if the first location is set

        # Main layout
        self.main_layout = QHBoxLayout(self)

        # Sidebar
        self.sidebar = QVBoxLayout()
        self.sidebar.setContentsMargins(10, 10, 10, 10)
        self.sidebar.setSpacing(10)
        self.sidebar_frame = QFrame(self)
        self.sidebar_frame.setStyleSheet("background-color: #2e2e2e; border-radius: 10px;")
        self.sidebar_frame.setLayout(self.sidebar)
        self.sidebar_frame.setFixedWidth(300)

        # Tabs - Vertical Style
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.West)
        self.tabs.setStyleSheet(
            "QTabWidget::pane { border: none; } "
            "QTabBar::tab { background-color: #2e2e2e; color: #fff; padding: 10px; margin: 2px; } "
            "QTabBar::tab:selected { background-color: #3e8e41; font-weight: medium; color: white; }"
        )
        self.home_tab = QWidget()
        self.tabs.addTab(self.home_tab, "Home")  # Removed Analytics tab

        # Set icon for the "Home" tab
        self.tabs.setTabIcon(0, QIcon(r"Utilis\Assets\home.png"))
        

        # Add the Home tab
        self.sidebar.addWidget(self.tabs)

        # Home Tab Layout
        self.setup_home_tab()

        # Map Widget
        self.map_view = QWebEngineView()
        self.create_map()

        # Add components to main layout
        self.main_layout.addWidget(self.sidebar_frame)
        self.main_layout.addWidget(self.map_view)

        # Buttons Layout (Bottom of Sidebar)
        self.bottom_button_layout = QVBoxLayout()

        # Clear Button
        self.clear_button = QPushButton()
        self.clear_button.setIcon(QIcon(r"Utilis\Assets\bin.png"))  # Replace with your file path for the plus icon
        self.clear_button.setIconSize(QSize(20, 20))
        self.clear_button.setStyleSheet("background-color: #3e8e41; color: white; font-size: 16px; padding: 10px;")
        self.clear_button.clicked.connect(self.clear_all)  # Connect to clear function
        self.bottom_button_layout.addWidget(self.clear_button)

        # See Graph Button
        self.graph_button = QPushButton()
        self.graph_button.setIcon(QIcon(r"Utilis\Assets\analysis.png"))  # Replace with your file path for the plus icon
        self.graph_button.setIconSize(QSize(20, 20))
        self.graph_button.setStyleSheet("background-color: #3e8e41; color: white; font-size: 16px; padding: 10px;")
        self.graph_button.clicked.connect(self.show_graph)  # Connect to graph function
        self.bottom_button_layout.addWidget(self.graph_button)

        # Add bottom button layout to the sidebar
        self.sidebar.addLayout(self.bottom_button_layout)

    def setup_home_tab(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)

        self.info_label = QLabel("Enter Latitude and Longitude:")
        self.info_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(self.info_label)

        self.lat_input = QTextEdit()
        self.lat_input.setReadOnly(False)
        self.lat_input.setPlaceholderText("Latitude")
        self.lat_input.setFixedHeight(40)
        layout.addWidget(self.lat_input)

        # Separator line after latitude
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.HLine)
        separator1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator1)

        
        self.lon_input = QTextEdit()
        self.lon_input.setReadOnly(False)
        self.lon_input.setPlaceholderText("Longitude")
        self.lon_input.setFixedHeight(40)
        layout.addWidget(self.lon_input)

        # Separator line after longitude
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator2)

        # Horizontal layout for buttons
        button_layout = QHBoxLayout()

        # Add Location Button with Icon
        self.add_location_button = QPushButton()
        self.add_location_button.setIcon(QIcon(r"Utilis\Assets\plus.png"))  # Replace with your file path for the plus icon
        self.add_location_button.setIconSize(QSize(20, 20))
        self.add_location_button.setStyleSheet("background-color: #3e8e41; color: #fff; font-size: 16px; padding: 10px;")
        self.add_location_button.clicked.connect(self.add_location)
        button_layout.addWidget(self.add_location_button)

        # Find Best Location Button with Icon
        self.find_button = QPushButton()
        self.find_button.setIcon(QIcon(r"Utilis\Assets\magnifying_glass.png"))  # Replace with your file path for the magnifying glass icon
        self.find_button.setIconSize(QSize(20, 20))
        self.find_button.setStyleSheet("background-color: #3e8e41; color: #fff; font-size: 16px; padding: 10px;")
        self.find_button.clicked.connect(self.find_best_location)
        self.find_button.setEnabled(False)  # Disable initially
        button_layout.addWidget(self.find_button)

        layout.addLayout(button_layout)  # Add the buttons horizontally

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("background-color: #2e2e2e; font-size: 14px;")
        layout.addWidget(self.result_text)

        self.home_tab.setLayout(layout)

    def create_map(self):
        # Create Folium map centered on Pakistan
        self.map = folium.Map(location=[30.3753, 69.3451], zoom_start=5, )

        # Highlight Pakistan boundary with neon-like glow using GeoJSON
        pakistan_boundary = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries/PAK.geo.json"
        
        folium.GeoJson(
            pakistan_boundary,
            name="Pakistan Boundary",
            style_function=lambda x: {
                "fillColor": "transparent",  # No fill
                "color": "green",  # Neon-like border color
                "weight": 2,  # Border thickness
                "opacity": 1,  # Full opacity for the glow effect
                "dashArray": "10, 5"  # Dotted or dashed line style
            },
            tooltip="Pakistan"  # Tooltip to show when hovering over the border
        ).add_to(self.map)

        # Save the map as an HTML file
        self.map_path = os.path.join(os.getcwd(), "map.html")
        self.map.save(self.map_path)

        # Load the map in QWebEngineView
        self.map_view.setUrl(QUrl.fromLocalFile(self.map_path))

    def update_map(self, lat, lon):
        # Add marker to the map for the given latitude and longitude
        folium.Marker(
            location=[lat, lon],
            popup=f"Latitude: {lat}, Longitude: {lon}",
            icon=folium.Icon(color="green")
        ).add_to(self.map)

        # Save updated map and reload it in QWebEngineView
        self.map.save(self.map_path)
        self.map_view.setUrl(QUrl.fromLocalFile(self.map_path))

    def add_location(self):
        try:
            # Get latitude and longitude from inputs
            lat = float(self.lat_input.toPlainText().strip())
            lon = float(self.lon_input.toPlainText().strip())

            if not self.is_first_location_set:
                # Assign values to the first location
                self.lat1, self.lon1 = lat, lon
                self.is_first_location_set = True
                self.result_text.append(f"First Location Set: Latitude {self.lat1}, Longitude {self.lon1}")
            else:
                # Assign values to the second location
                self.lat2, self.lon2 = lat, lon
                self.result_text.append(f"Second Location Set: Latitude {self.lat2}, Longitude {self.lon2}")
                self.find_button.setEnabled(True)  # Enable the "Find Best Location" button

            # Update the map with the new marker
            self.update_map(lat, lon)

            # Clear input fields
            self.lat_input.clear()
            self.lon_input.clear()

        except ValueError:
            self.result_text.setText("Invalid latitude or longitude. Please enter valid numerical values.")

    def generate_parameters(self, latitude, longitude, start_date, end_date):
        # Generate dates for the next 20 days starting from the specified start date
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        input_data_list = []
        for date in date_range:
            current_hour = 12  # Using noon as the representative hour for daily average
            current_day_of_week = date.weekday()  # 0=Monday, 6=Sunday
            current_month = date.month
            current_season = (current_month % 12 + 3) // 3  # Calculate season based on the month

            input_data_list.append({
                'latitude': latitude,
                'longitude': longitude,
                'hour': current_hour,
                'day_of_week': current_day_of_week,
                'month': current_month,
                'season': current_season,
            })

        input_data = pd.DataFrame(input_data_list)

        # Predict parameters using the trained model for each future date
        predicted_parameters = param_model.predict(input_data)

        predicted_df = pd.DataFrame(predicted_parameters, columns=[ 
            'ghi_pyr', 'dni', 'dhi', 'air_temperature', 'relative_humidity',
            'wind_speed', 'wind_speed_of_gust', 'wind_from_direction', 'barometric_pressure'
        ])
        predicted_df['date'] = date_range

        return predicted_df

    def predict_power_for_location(self, predicted_params_df):
        # Use the predicted parameters to predict the solar power output
        power_predictions = power_model.predict(predicted_params_df.drop(columns=['date']))  # Drop date for prediction

        predicted_params_df['predicted_solar_power'] = power_predictions

        return predicted_params_df

    def find_best_location(self):
        if self.lat1 is None or self.lon1 is None or self.lat2 is None or self.lon2 is None:
            self.result_text.setText("Please enter valid locations first.")
            return

        start_date = datetime.today()
        end_date = start_date + timedelta(days=400)

        # Generate parameters for both locations
        location1_params = self.generate_parameters(self.lat1, self.lon1, start_date, end_date)
        location2_params = self.generate_parameters(self.lat2, self.lon2, start_date, end_date)

        # Predict solar power for both locations
        location1_power = self.predict_power_for_location(location1_params)
        location2_power = self.predict_power_for_location(location2_params)


        # Store the power data in the class attributes
        self.location1_power = location1_power
        self.location2_power = location2_power

        avg_power_location1 = location1_power['predicted_solar_power'].mean()
        print(avg_power_location1)
        avg_power_location2 = location2_power['predicted_solar_power'].mean()
        print(avg_power_location2)
        # Compare the average solar power for both locations
        if avg_power_location1 > avg_power_location2:
            best_location = f"\n\nLocation 1 (Lat: {self.lat1}, Lon: {self.lon1}) with average power {avg_power_location1:.2f} W/m^2."
        else:
            best_location = f"\n\nLocation 2 (Lat: {self.lat2}, Lon: {self.lon2}) with average power {avg_power_location2:.2f} W/m^2."

        self.result_text.setText(f"The best location for solar power generation is: \n\n{best_location}")

    

    def clear_all(self):
        # Reset inputs and variables
        self.lat_input.clear()
        self.lon_input.clear()
        self.result_text.clear()
        self.is_first_location_set = False
        self.lat1 = self.lon1 = self.lat2 = self.lon2 = None
        self.find_button.setEnabled(False)  # Disable the "Find Best Location" button
        self.create_map()  # Recreate the map (reset to initial state)
        self.map_view.setUrl(QUrl.fromLocalFile(self.map_path))



    def plot_2d_power_graph(self, location1_power, location2_power, degree=3):
        # Create a figure for 2D plotting
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create a range of dates for the x-axis (assuming 'date' is a list of datetime objects)
        dates = pd.to_datetime(location2_power['date'])
        dates_numeric = mdates.date2num(dates)  # Convert dates to numeric format for fitting

        # For Y, use the predicted solar power from both locations
        power_loc1_values = location1_power['predicted_solar_power'].values
        power_loc2_values = location2_power['predicted_solar_power'].values

        # Fit polynomial curves to the data (default degree=3 for cubic fit)
        loc1_coeffs = np.polyfit(dates_numeric, power_loc1_values, degree)
        loc2_coeffs = np.polyfit(dates_numeric, power_loc2_values, degree)

        # Generate fitted curves
        x_smooth = np.linspace(dates_numeric.min(), dates_numeric.max(), 1000)
        loc1_fitted = np.polyval(loc1_coeffs, x_smooth)
        loc2_fitted = np.polyval(loc2_coeffs, x_smooth)

        # Convert x_smooth back to datetime for plotting
        dates_smooth = mdates.num2date(x_smooth)

        # Plot the fitted curves
        ax.plot(dates_smooth, loc1_fitted, label='Location 1 (Fitted)', color='b')
        ax.plot(dates_smooth, loc2_fitted, label='Location 2 (Fitted)', color='g')

        # Optionally plot the original data points for comparison
        ax.scatter(dates, power_loc1_values, label='Location 1 (Data)', color='b', alpha=0.5)
        ax.scatter(dates, power_loc2_values, label='Location 2 (Data)', color='g', alpha=0.5)

        # Set the y-axis range from 0 to 900
        ax.set_ylim(0, 900)

        # Add labels and title
        ax.set_xlabel('Month')
        ax.set_ylabel('Predicted Solar Power (W/m^2)')
        ax.set_title('2D Predicted Solar Power with Polynomial Curve Fitting')

        # Format the x-axis to show month names
        ax.xaxis.set_major_locator(mdates.MonthLocator())  # Place a tick at the start of each month
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Format the ticks as month names (e.g., Jan, Feb)

        # Rotate the x-axis labels to make them readable
        plt.xticks(rotation=45, ha='right')

        # Add a legend to differentiate the locations
        ax.legend()

        # Enable grid for better visualization
        ax.grid(True)

        # Allow interactive features like zooming and panning
        plt.tight_layout()

        return fig  # Return the figure object




    def show_graph(self):
        # Create the popup dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Solar Power Graph")
        dialog.setFixedSize(1000, 1000)  # Set a fixed size for the popup
        
        # Create a layout for the dialog
        layout = QVBoxLayout()

        # Generate the graph
        fig = self.plot_2d_power_graph(self.location1_power, self.location2_power)

        # Create the FigureCanvas and add it to the dialog layout
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        # Set the layout of the dialog
        dialog.setLayout(layout)

        # Show the dialog
        dialog.exec_()



# Create the application and run
app = QApplication(sys.argv)
window = SolarParkGUI()
window.show()
sys.exit(app.exec_())
