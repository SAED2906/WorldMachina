#!/usr/bin/python3
import numpy as np
import os
import sys
import util
import argparse
import multiprocessing
from PIL import Image
from numba import jit
import time
from scipy.ndimage import zoom
import math
import logging
import tempfile
import shutil
import gc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('erosion')

@jit(nopython=True)
def apply_slippage_optimized(terrain, delta, smoothed, repose_slope):
    result = np.copy(terrain)
    for i in range(terrain.shape[0]):
        for j in range(terrain.shape[1]):
            if np.abs(delta[i, j]) > repose_slope:
                result[i, j] = smoothed[i, j]
    return result

def apply_slippage(terrain, repose_slope, cell_width):
    delta = util.simple_gradient(terrain) / cell_width
    smoothed = util.gaussian_blur(terrain, sigma=1.5)
    return apply_slippage_optimized(terrain, delta, smoothed, repose_slope)

def make_square(terrain):
    height, width = terrain.shape
    max_dim = max(height, width)
    square_terrain = np.zeros((max_dim, max_dim), dtype=terrain.dtype)
    y_offset = (max_dim - height) // 2
    x_offset = (max_dim - width) // 2
    square_terrain[y_offset:y_offset+height, x_offset:x_offset+width] = terrain
    return square_terrain

def safe_fbm(shape, p, lower=-np.inf, upper=np.inf):
    max_dim = max(shape)
    square_shape = (max_dim, max_dim)
    square_noise = util.fbm(square_shape, p, lower, upper)
    if shape[0] == shape[1]:
        return square_noise
    else:
        y_offset = (max_dim - shape[0]) // 2
        x_offset = (max_dim - shape[1]) // 2
        return square_noise[y_offset:y_offset+shape[0], x_offset:x_offset+shape[1]]

def process_tile(args):
    tile_data, tile_id, full_width, scale_factor, overlap, iterations, verbose, swap_dir = args
    
    tile_height, tile_width = tile_data.shape
    cell_width = full_width / (tile_width * scale_factor)
    cell_area = cell_width ** 2
    
    rain_rate = 0.0006 * cell_area
    evaporation_rate = 0.0004
    min_height_delta = 0.03
    repose_slope = 0.02
    gravity = 35.0
    sediment_capacity_constant = 60.0
    dissolving_rate = 0.3
    deposition_rate = 0.0008
    
    terrain = np.copy(tile_data)
    sediment = np.zeros_like(terrain)
    water = np.zeros_like(terrain)
    velocity = np.zeros_like(terrain)
    
    shape = terrain.shape
    
    if verbose:
        logger.info(f'Processing tile {tile_id}, shape: {shape}')
    
    tile_iterations = int(iterations * math.sqrt(scale_factor) * 1.5)
    print_interval = max(1, tile_iterations // 20)
    
    for i in range(tile_iterations):
        if verbose:
            logger.info(f'Tile {tile_id}: Iteration {i+1}/{tile_iterations} ({(i+1)/tile_iterations*100:.1f}%)')

        water += (0.9 + 0.2 * np.random.rand()) * np.random.rand(*shape) * rain_rate

        gradient = util.simple_gradient(terrain)
        gradient_magnitude = np.abs(gradient)
        mask = gradient_magnitude < 1e-10
        random_directions = np.exp(2j * np.pi * np.random.rand(*shape))
        gradient = np.where(mask, random_directions, gradient / gradient_magnitude)

        neighbor_height = util.sample(terrain, -gradient)
        height_delta = terrain - neighbor_height
        
        sediment_capacity = (
            (np.maximum(height_delta, min_height_delta) / cell_width) * 
            velocity * water * sediment_capacity_constant
        )
        
        deposited_sediment = np.where(
            height_delta < 0,
            np.minimum(height_delta, sediment),
            np.where(
                sediment > sediment_capacity,
                deposition_rate * (sediment - sediment_capacity),
                dissolving_rate * (sediment - sediment_capacity)
            )
        )

        deposited_sediment = np.maximum(-height_delta, deposited_sediment)

        sediment -= deposited_sediment
        terrain += deposited_sediment
        sediment = util.displace(sediment, gradient)
        water = util.displace(water, gradient)

        terrain = apply_slippage(terrain, repose_slope, cell_width)

        velocity = gravity * height_delta / cell_width
      
        water *= 1 - (evaporation_rate * (0.9 + 0.2 * np.random.rand()))
    
    if swap_dir:
        tile_file = os.path.join(swap_dir, f"tile_{tile_id}.npy")
        np.save(tile_file, terrain)
        return tile_file, tile_id
    else:
        return terrain, tile_id

def split_into_tiles(heightmap, tile_size, overlap, scale_factor, max_memory_gb=None):
    height, width = heightmap.shape
    
    if max_memory_gb:
        pixel_size_bytes = heightmap.itemsize
        total_pixels = height * width * (scale_factor ** 2)
        total_size_gb = total_pixels * pixel_size_bytes / (1024**3)
        
        if total_size_gb > max_memory_gb:
            memory_ratio = max_memory_gb / total_size_gb
            size_factor = math.sqrt(memory_ratio) * 0.5
            logger.info(f"Memory constraint: {max_memory_gb}GB. Estimated data size: {total_size_gb:.2f}GB")
            logger.info(f"Adjusting tile size by factor: {size_factor:.2f}")
            
            tile_size = min(tile_size, int(tile_size * size_factor))
            tile_size = max(256, tile_size)
    
    adjusted_tile_size = min(tile_size, int(tile_size / math.sqrt(scale_factor)))
    adjusted_tile_size = max(256, adjusted_tile_size)
    
    adjusted_overlap = min(overlap, int(adjusted_tile_size * 0.2))
    
    n_tiles_y = max(1, int(np.ceil(height / (adjusted_tile_size - adjusted_overlap))))
    n_tiles_x = max(1, int(np.ceil(width / (adjusted_tile_size - adjusted_overlap))))
    
    tiles = []
    tile_positions = []
    
    for i in range(n_tiles_y):
        for j in range(n_tiles_x):
            start_y = max(0, i * (adjusted_tile_size - adjusted_overlap))
            end_y = min(height, start_y + adjusted_tile_size)
            
            start_x = max(0, j * (adjusted_tile_size - adjusted_overlap))
            end_x = min(width, start_x + adjusted_tile_size)
            
            tile = heightmap[start_y:end_y, start_x:end_x]
            
            if tile.shape[0] != tile.shape[1]:
                max_dim = max(tile.shape)
                square_tile = np.zeros((max_dim, max_dim), dtype=tile.dtype)
                y_offset = (max_dim - tile.shape[0]) // 2
                x_offset = (max_dim - tile.shape[1]) // 2
                square_tile[y_offset:y_offset+tile.shape[0], x_offset:x_offset+tile.shape[1]] = tile
                tile = square_tile
                end_y = start_y + tile.shape[0]
                end_x = start_x + tile.shape[1]
            
            tiles.append(tile)
            tile_positions.append((start_y, end_y, start_x, end_x))
    
    return tiles, tile_positions, n_tiles_y, n_tiles_x, adjusted_tile_size, adjusted_overlap

def stitch_tiles_from_disk(tile_files, tile_positions, original_shape, n_tiles_y, n_tiles_x, overlap, swap_dir):
    result = np.zeros(original_shape, dtype=float)
    weights = np.zeros_like(result)
    
    for i, ((tile_file, _), (start_y, end_y, start_x, end_x)) in enumerate(zip(tile_files, tile_positions)):
        tile = np.load(tile_file)
        
        tile_height, tile_width = tile.shape
        
        end_y = min(start_y + tile_height, original_shape[0])
        end_x = min(start_x + tile_width, original_shape[1])
        
        tile_cropped = tile[:end_y-start_y, :end_x-start_x]
        
        weight_mask = np.ones((end_y - start_y, end_x - start_x))
        
        if start_y > 0:
            for k in range(min(overlap, end_y - start_y)):
                factor = ((k + 1) / (overlap + 1)) ** 2
                weight_mask[k, :] = factor
                    
        if end_y < original_shape[0]:
            for k in range(min(overlap, end_y - start_y)):
                factor = ((k + 1) / (overlap + 1)) ** 2
                weight_mask[-(k+1), :] = factor
                    
        if start_x > 0:
            for k in range(min(overlap, end_x - start_x)):
                factor = ((k + 1) / (overlap + 1)) ** 2
                weight_mask[:, k] = factor
                    
        if end_x < original_shape[1]:
            for k in range(min(overlap, end_x - start_x)):
                factor = ((k + 1) / (overlap + 1)) ** 2
                weight_mask[:, -(k+1)] = factor
        
        result[start_y:end_y, start_x:end_x] += tile_cropped * weight_mask
        weights[start_y:end_y, start_x:end_x] += weight_mask
        
        del tile
        del tile_cropped
        if i % 10 == 0:
            gc.collect()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(weights > 0, result / weights, 0)
    
    return result

def stitch_tiles(processed_tiles, tile_positions, original_shape, n_tiles_y, n_tiles_x, overlap):
    result = np.zeros(original_shape, dtype=float)
    weights = np.zeros_like(result)
    
    for (tile, (start_y, end_y, start_x, end_x)) in zip(processed_tiles, tile_positions):
        tile_height, tile_width = tile.shape
        
        end_y = min(start_y + tile_height, original_shape[0])
        end_x = min(start_x + tile_width, original_shape[1])
        
        tile_cropped = tile[:end_y-start_y, :end_x-start_x]
        
        weight_mask = np.ones((end_y - start_y, end_x - start_x))
        
        if start_y > 0:
            for i in range(min(overlap, end_y - start_y)):
                factor = ((i + 1) / (overlap + 1)) ** 2
                weight_mask[i, :] = factor
                    
        if end_y < original_shape[0]:
            for i in range(min(overlap, end_y - start_y)):
                factor = ((i + 1) / (overlap + 1)) ** 2
                weight_mask[-(i+1), :] = factor
                    
        if start_x > 0:
            for i in range(min(overlap, end_x - start_x)):
                factor = ((i + 1) / (overlap + 1)) ** 2
                weight_mask[:, i] = factor
                    
        if end_x < original_shape[1]:
            for i in range(min(overlap, end_x - start_x)):
                factor = ((i + 1) / (overlap + 1)) ** 2
                weight_mask[:, -(i+1)] = factor
        
        result[start_y:end_y, start_x:end_x] += tile_cropped * weight_mask
        weights[start_y:end_y, start_x:end_x] += weight_mask
    
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(weights > 0, result / weights, 0)
    
    return result

def estimate_memory_usage(shape, scale_factor, tile_size, overlap):
    height, width = shape
    scaled_shape = (int(height * scale_factor), int(width * scale_factor))
    adjusted_tile_size = min(tile_size, int(tile_size / math.sqrt(scale_factor)))
    
    n_tiles_y = max(1, int(np.ceil(height / (adjusted_tile_size - overlap))))
    n_tiles_x = max(1, int(np.ceil(width / (adjusted_tile_size - overlap))))
    
    total_tiles = n_tiles_y * n_tiles_x
    
    bytes_per_pixel = 4
    
    tile_bytes = (adjusted_tile_size ** 2) * bytes_per_pixel * 4
    
    result_bytes = scaled_shape[0] * scaled_shape[1] * bytes_per_pixel * 2
    
    total_bytes = tile_bytes * total_tiles + result_bytes
    
    return total_bytes / (1024**3)

def main():
    parser = argparse.ArgumentParser(description='High-resolution planetary erosion simulation')
    parser.add_argument('input_file', help='Input heightmap file (image or numpy array)')
    parser.add_argument('--output', '-o', default='detailed_erosion', help='Output filename prefix')
    parser.add_argument('--scale', '-s', type=float, default=4.0, help='Scale factor for resolution')
    parser.add_argument('--tile-size', '-t', type=int, default=1024, help='Base size of individual tiles')
    parser.add_argument('--overlap', type=int, default=192, help='Base overlap between tiles')
    parser.add_argument('--processes', '-p', type=int, default=None, help='Number of processes')
    parser.add_argument('--iterations', '-i', type=int, default=None, help='Base erosion iterations')
    parser.add_argument('--planet-diameter', '-d', type=float, default=12742.0, help='Planet diameter in km')
    parser.add_argument('--force-square', action='store_true', help='Force square input')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal logging')
    parser.add_argument('--swap', action='store_true', help='Use disk swapping for large-scale processing')
    parser.add_argument('--swap-dir', type=str, default=None, help='Directory for temporary swap files')
    parser.add_argument('--max-memory', type=float, default=None, help='Maximum memory usage in GB')
    parser.add_argument('--local-swap', action='store_true', help='Use script directory for swap files')
    args = parser.parse_args()
    
    if args.quiet:
        logger.setLevel(logging.WARNING)
    elif args.verbose:
        logger.setLevel(logging.DEBUG)
    
    start_time = time.time()
    full_width = args.planet_diameter * np.pi
    
    logger.info(f"Starting erosion simulation with scale factor: {args.scale}x")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    swap_dir = None
    if args.swap:
        if args.swap_dir:
            swap_dir = args.swap_dir
        elif args.local_swap:
            swap_dir = os.path.join(script_dir, f"erosion_swap_{int(time.time())}")
            logger.info(f"Using local directory for swap files")
        else:
            swap_dir = tempfile.mkdtemp(prefix="erosion_swap_")
        
        os.makedirs(swap_dir, exist_ok=True)
        logger.info(f"Using disk swapping in directory: {swap_dir}")
    
    try:
        logger.info(f"Loading heightmap from {args.input_file}...")
        
        try:
            terrain, _ = util.load_from_file(args.input_file)
        except:
            heightmap_img = Image.open(args.input_file).convert('L')
            terrain = np.array(heightmap_img, dtype=float)
            terrain = util.normalize(terrain)
        
        logger.info(f"Original heightmap shape: {terrain.shape}")
        
        estimated_memory = estimate_memory_usage(terrain.shape, args.scale, args.tile_size, args.overlap)
        logger.info(f"Estimated memory usage: {estimated_memory:.2f} GB")
        
        if args.max_memory and estimated_memory > args.max_memory and not args.swap:
            logger.warning(f"Estimated memory ({estimated_memory:.2f} GB) exceeds maximum ({args.max_memory} GB)")
            logger.warning("Enabling disk swapping automatically")
            args.swap = True
            if not swap_dir:
                swap_dir = tempfile.mkdtemp(prefix="erosion_swap_")
                logger.info(f"Using disk swapping in directory: {swap_dir}")
        
        height, width = terrain.shape
        aspect_ratio = width / height
        
        if args.force_square or aspect_ratio != 1.0:
            logger.info(f"Making image square (aspect ratio: {aspect_ratio:.2f})...")
            terrain = make_square(terrain)
            logger.info(f"New shape: {terrain.shape}")
        
        tile_size = args.tile_size
        overlap = args.overlap
        
        logger.info(f"Splitting into tiles with base size {tile_size}x{tile_size}...")
        tiles, tile_positions, n_tiles_y, n_tiles_x, adjusted_tile_size, adjusted_overlap = split_into_tiles(
            terrain, tile_size, overlap, args.scale, args.max_memory
        )
        logger.info(f"Created {len(tiles)} tiles ({n_tiles_x}x{n_tiles_y} grid) with adjusted size {adjusted_tile_size} and overlap {adjusted_overlap}")
        
        if args.swap:
            del terrain
            gc.collect()
        
        if args.scale != 1.0:
            logger.info(f"Scaling tiles by factor {args.scale}...")
            scaled_tiles = []
            scaled_positions = []
            
            for i, (tile, (start_y, end_y, start_x, end_x)) in enumerate(zip(tiles, tile_positions)):
                scaled_tile = zoom(tile, args.scale, order=3)
                
                new_start_y = int(start_y * args.scale)
                new_end_y = int(end_y * args.scale)
                new_start_x = int(start_x * args.scale)
                new_end_x = int(end_x * args.scale)
                
                if args.scale > 2.0:
                    noise_scale = 0.05
                    tile_shape = scaled_tile.shape
                    noise = safe_fbm(tile_shape, -1.5)
                    scaled_tile = scaled_tile * (1.0 - noise_scale) + noise * noise_scale
                
                scaled_tiles.append(scaled_tile)
                scaled_positions.append((new_start_y, new_end_y, new_start_x, new_end_x))
                
                if args.verbose or (i % max(1, len(tiles) // 10) == 0):
                    logger.info(f"Scaled {i+1}/{len(tiles)} tiles")
            
            tiles = scaled_tiles
            tile_positions = scaled_positions
            
            original_shape = (int(height * args.scale), int(width * args.scale))
            logger.info(f"New scaled shape: {original_shape}")
        else:
            original_shape = terrain.shape
        
        if 'scaled_tiles' in locals():
            del scaled_tiles 
        else:
            del tiles

        gc.collect()
        
        if args.iterations is None:
            iterations = int(2.0 * math.sqrt(original_shape[0] * original_shape[1] / 1e6))
            iterations = max(100, min(iterations, 800))
        else:
            iterations = args.iterations
            
        logger.info(f"Running {iterations} base erosion iterations per tile")
        
        num_processes = args.processes if args.processes is not None else multiprocessing.cpu_count()
        logger.info(f"Processing tiles using {num_processes} processes")
        
        process_args = [
            (tiles[i], i, full_width, args.scale, adjusted_overlap, iterations, args.verbose, swap_dir)
            for i in range(len(tiles))
        ]
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(process_tile, process_args)
        
        del tiles
        gc.collect()
        
        logger.info("Stitching tiles back together...")
        if args.swap:
            stitched_terrain = stitch_tiles_from_disk(results, tile_positions, original_shape, 
                                                     n_tiles_y, n_tiles_x, adjusted_overlap, swap_dir)
        else:
            processed_tiles = [None] * len(tile_positions)
            for tile, tile_id in results:
                processed_tiles[tile_id] = tile
            
            stitched_terrain = stitch_tiles(processed_tiles, tile_positions, original_shape, 
                                         n_tiles_y, n_tiles_x, adjusted_overlap)
            
            del processed_tiles
            gc.collect()
        
        stitched_terrain = util.normalize(stitched_terrain)
        
        logger.info("Applying final detail enhancements...")
        stitched_terrain = util.gaussian_blur(stitched_terrain, sigma=0.5)
        
        output_file = f"{args.output}_scale{args.scale:.1f}"
        np.save(f"{output_file}.npy", stitched_terrain)
        
        logger.info(f"Saving results to {output_file}.png and {output_file}.npy...")
        image_output = Image.fromarray((stitched_terrain * 255).astype(np.uint8))
        image_output.save(f"{output_file}.png", quality=95, optimize=True)
        
        try:
            logger.info("Generating hillshaded visualization...")
            hillshaded = util.hillshaded(stitched_terrain)
            hillshade_img = Image.fromarray((hillshaded * 255).astype(np.uint8))
            hillshade_img.save(f"{output_file}_hillshade.png", quality=95, optimize=True)
        except Exception as e:
            logger.warning(f"Could not create hillshade: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        
        logger.info(f"Simulation complete in {minutes}m {seconds}s")
        logger.info(f"Output shape: {stitched_terrain.shape} (scale: {args.scale}x)")
        logger.info(f"Files saved: {output_file}.png and {output_file}.npy")
        if os.path.exists(f"{output_file}_hillshade.png"):
            logger.info(f"Hillshade visualization: {output_file}_hillshade.png")
        
    except Exception as e:
        logger.error(f"Error in simulation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if args.swap and swap_dir and not args.swap_dir:
            try:
                logger.info(f"Cleaning up temporary swap directory: {swap_dir}")
                shutil.rmtree(swap_dir)
            except Exception as e:
                logger.warning(f"Error cleaning up swap directory: {e}")

    return 0

if __name__ == '__main__':
    sys.exit(main())