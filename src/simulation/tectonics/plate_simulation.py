import platec
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import logging
from PIL import Image, ImageDraw
import colorsys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

constant = 2
width = 1024 * constant
height = 1024 * constant

seed = 2
sea_level = 0.25
erosion_period = 60
folding_ratio = 0.015
aggr_overlap_abs = 1000000
aggr_overlap_rel = 0.7
cycle_count = 4
num_plates = 12

logger.info(f"Starting simulation with seed={seed}, dimensions={width}x{height}, num_plates={num_plates}")

start_time = time.time()
litho = platec.create(seed, width, height, sea_level, erosion_period,
                     folding_ratio, aggr_overlap_abs, aggr_overlap_rel,
                     cycle_count, num_plates)
logger.info(f"Lithosphere initialized in {time.time() - start_time:.2f} seconds")

max_steps = 10000
logger.info(f"Running simulation for up to {max_steps} steps")

step_start_time = time.time()
for i in range(max_steps):
    if i % 50 == 0 and i > 0:
        elapsed = time.time() - step_start_time
        steps_per_second = i/elapsed if elapsed > 0 else 0
        logger.info(f"Completed {i} steps ({steps_per_second:.1f} steps/second)")
        
        hm = platec.get_heightmap(litho)
        hm_array = np.array(hm)
        logger.info(f"Heightmap stats - min: {np.min(hm_array):.4f}, max: {np.max(hm_array):.4f}, mean: {np.mean(hm_array):.4f}")
        
    if platec.is_finished(litho):
        logger.info(f"Simulation finished after {i+1} steps")
        break
    platec.step(litho)

total_time = time.time() - start_time
logger.info(f"Simulation completed in {total_time:.2f} seconds")

logger.info("Retrieving final heightmap")
heightmap = platec.get_heightmap(litho)
heightmap_2d = np.array(heightmap).reshape(height, width)

logger.info("Retrieving plates map")
platesmap = platec.get_platesmap(litho)
platesmap_2d = np.array(platesmap).reshape(height, width)

logger.info(f"Final heightmap - min: {np.min(heightmap_2d):.4f}, max: {np.max(heightmap_2d):.4f}, mean: {np.mean(heightmap_2d):.4f}")
unique_plates = np.unique(platesmap_2d)
logger.info(f"Number of unique plates: {len(unique_plates)}")
logger.info(f"Plate IDs: {unique_plates}")

logger.info("Destroying lithosphere")
platec.destroy(litho)

logger.info("Saving raw heightmap data")
normalized_heightmap_16bit = ((heightmap_2d - np.min(heightmap_2d)) / 
                             (np.max(heightmap_2d) - np.min(heightmap_2d)) * 65535).astype(np.uint16)
raw_image = Image.fromarray(normalized_heightmap_16bit)
raw_output_file = 'heightmap_raw_16bit.png'
raw_image.save(raw_output_file)
logger.info(f"Saved raw heightmap to {raw_output_file}")

normalized_heightmap_8bit = ((heightmap_2d - np.min(heightmap_2d)) / 
                            (np.max(heightmap_2d) - np.min(heightmap_2d)) * 255).astype(np.uint8)
raw_8bit_image = Image.fromarray(normalized_heightmap_8bit)
raw_8bit_output_file = 'heightmap_raw_8bit.png'
raw_8bit_image.save(raw_8bit_output_file)
logger.info(f"Saved 8-bit heightmap to {raw_8bit_output_file}")

logger.info("Creating colored terrain visualization")

def get_terrain_color(height, sea_level=0.6):
    if height < sea_level - 0.2:
        return (0, 0, 128)
    elif height < sea_level - 0.05:
        depth_ratio = (height - (sea_level - 0.2)) / 0.15
        return (0, min(255, int(64 + depth_ratio * 64)), min(255, 180))
    elif height < sea_level:
        depth_ratio = (height - (sea_level - 0.05)) / 0.05
        return (0, min(255, int(128 + depth_ratio * 64)), min(255, 255))
    elif height < sea_level + 0.05:
        return (min(255, 240), min(255, 220), min(255, 160))
    elif height < sea_level + 0.15:
        return (min(255, 30), min(255, 180 - int((height - sea_level - 0.05) * 300)), min(255, 30))
    elif height < sea_level + 0.3:
        return (min(255, 140), min(255, 100), min(255, 40))
    elif height < sea_level + 0.5:
        return (min(255, 120 + int((height - sea_level - 0.3) * 200)), 
                min(255, 100 - int((height - sea_level - 0.3) * 150)),
                min(255, 80 - int((height - sea_level - 0.3) * 100)))
    else:
        whiteness = min(255, int(((height - (sea_level + 0.5)) / 0.2) * 255))
        return (min(255, 200 + whiteness // 4), 
                min(255, 200 + whiteness // 4), 
                min(255, 210 + whiteness // 4))

height_min = np.min(heightmap_2d)
height_max = np.max(heightmap_2d)
normalized_height = (heightmap_2d - height_min) / (height_max - height_min)

rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
for y in range(height):
    for x in range(width):
        rgb_array[y, x] = get_terrain_color(normalized_height[y, x], sea_level)

terrain_image = Image.fromarray(rgb_array, 'RGB')
terrain_output_file = 'terrain_colored.png'
terrain_image.save(terrain_output_file)
logger.info(f"Saved colored terrain to {terrain_output_file}")

logger.info("Creating matplotlib visualizations")
plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
img1 = plt.imshow(heightmap_2d, cmap='terrain')
plt.colorbar(img1, label='Elevation')
plt.title('Raw Terrain Heightmap')

plt.subplot(2, 2, 3)
img3 = plt.imshow(platesmap_2d, cmap='tab10')
plt.colorbar(img3, label='Plate Index')
plt.title('Tectonic Plates')

plt.subplot(2, 2, 4)
img4 = plt.imshow(terrain_image)
plt.title('Colored Terrain')

plt.tight_layout()
output_file = 'terrain_visualization.png'
logger.info(f"Saving visualization to {output_file}")
plt.savefig(output_file, dpi=300)

logger.info("Done!")