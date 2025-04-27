# WorldMachina: Physically Accurate Planet Generation & Visualization

PlanetViewer is an interactive 3D application for generating and visualizing physically accurate planetary terrain using tectonic plate simulation. Perfect for my TTRPG world-building (I hope), scientific visualization, or anyone interested in procedural planet generation.

## Features

WorldMachina currently offers real-time 3D planet visualization with OpenGL, letting you explore planetary terrain with adjustable parameters. I've built heightmap-based terrain rendering to make your planets look detailed and realistic. The ocean comes with configurable sea levels so you can find the params that match your map. I've included a skybox background system to place your planet in whatever scene suits you. The tectonic plate simulation through PyPlatec generates realistic terrain based on actual geological principles.

## Getting Started

### Prerequisites

WorldMachina needs Python 3.7+, along with these libraries: PyGame for window management and 2D UI, PyOpenGL for 3D rendering, NumPy for numerical operations, Matplotlib for visualization, PIL/Pillow for image processing, and PyPlatec for the tectonic plate simulation.

### Installation

First, clone the repository to your computer. I strongly recommend setting up a Python virtual environment to keep dependencies isolated and avoid conflicts with other projects:

```bash
git https://github.com/SAED2906/WorldMachina.git
cd WorldMachina

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

Once your virtual environment is activated, you can run the installation script I created:

```bash
# Make script executable (Linux/Mac)
chmod +x install.sh
# Run the script
./install.sh
```

Or install dependencies manually:

```bash
pip install pygame pyopengl numpy matplotlib pillow
CXXFLAGS="-std=c++14" pip install PyPlatec
```

The PyPlatec installation requires C++14 support, which is why we use the CXXFLAGS environment variable.

### Running PlanetViewer

Launch the application by running the main script (make sure your virtual environment is activated):

```bash
python src/main.py
```

## Usage

WorldMachina has straightforward controls for exploring your planet. 

For camera control:
- Hold left mouse button and drag to rotate the camera around the planet
- Use the mouse scroll wheel to zoom in and out
- The camera will always focus on the center of the planet

For display options:
- Press 'S' to open the skybox selector where you can choose different space backgrounds
- Press 'F11' to toggle fullscreen mode
- Press 'ESC' to exit the application

For terrain visualization:
- Press 'N' to toggle normal mapping (enhances surface detail) (Not added)
- Press 'B' to toggle bump mapping (adds small surface irregularities) (Not added)
- Press 'H' to toggle height mapping (applies the heightmap displacement)
- Use '+' and '-' keys to increase or decrease bump strength (Not added)
- Use '[' and ']' keys to modify height scale (makes mountains taller or shorter)

For ocean parameters:
- Use UP/DOWN arrow keys to raise or lower the sea level
- Use 'O' and 'P' keys to increase or decrease the ocean radius

The UI panels on the left and right sides of the screen show all controls and current settings for easy reference. The right panel displays information about the current skybox if one is selected, as well as camera positioning data.

## Terrain Generation

I've integrated PyPlatec for tectonic plate simulation. This creates terrain heightmaps based on how tectonic plates move and interact. The code in `src/simulation/tectonics/plate_simulation.py` runs the plate tectonic generation, creating a heightmap that shows up as both raw elevation data and colored terrain. You can tweak parameters like plate count, sea level, and erosion period to get different results.

## Project Structure

I've organized the project with clear separation between components. The main entry point is `src/main.py`, which sets up the engine and window. The core rendering happens in `src/window.py`, handling the OpenGL context and user input.

In the rendering system (`src/render/`), I've created modules for planet rendering (`planet.py`), skybox backgrounds (`skybox.py`), and shader management (`shader_manager.py`). These work with the GLSL shaders in `src/shaders/` to create heightmap displacement, ocean effects, and lighting.

The UI components in `src/ui/` handle things like the texture selector. The simulation code in `src/simulation/tectonics/` powers the plate tectonic generation.

## Roadmap

## Roadmap

Future development plans include enhancing the erosion simulation to create more realistic terrain weathering effects like river valleys, deltas, and coastline features. I'm currently working on adding fractal detail to large-scale tectonic features for more natural-looking terrain at various zoom levels, and implementing different rock types and terrain properties based on formation processes.

Building on the early climate simulation work I've started, I plan to develop a full global climate system with temperature, pressure, and wind patterns. This will include rainfall calculation that properly considers topography, creating realistic rain shadows behind mountains. The climate system will ultimately generate biome classifications based on temperature and precipitation values, allowing for realistic environmental distribution across the planet. I also want to create atmospheric rendering effects that vary with the climate conditions.

A comprehensive water system is high on my priority list, with river networks that follow terrain contours and realistic lake formation in terrain depressions. I'm planning to add dynamic sea levels with coastal features like beaches, cliffs, and estuaries, as well as glacier and ice cap systems for polar and high-elevation regions. These hydrological features will work together with the climate system to create coherent and believable planetary environments.

Visualization improvements will make the planets more immersive and detailed. This includes adding vegetation rendering based on biome types, implementing dynamic time-of-day lighting and season simulation, creating atmospheric scattering effects for realistic skies and horizons, and adding cloud systems that reflect the underlying climate patterns.

For better usability, I want to build a more comprehensive UI for parameter adjustment and feature selection. I plan to add the ability to save, load, and share planet configurations, create export options for heightmaps, climate data, and visualization frames, and develop annotation and measurement tools that would be particularly useful for world-building and geography applications.

## Support the Project

If you find this project useful for your TTRPG world-building, scientific visualization, or other creative endeavors, consider supporting the development:

https://buymeacoffee.com/williammarais

## About

I created this project during my computer science studies and out of my love for tabletop role-playing games. While developing a custom Pathfinder module, I wanted a scientifically accurate world to base it on, which led me to *try* build this physically-based planet generator.

## License

This project is licensed under the MIT License - see the LICENSE file for details.