import os 
import time
import subprocess
from tqdm import tqdm
import numpy as np
import rasterio
from rasterio.windows import Window


def gfill_with_data(fi, fi_fill, esa, fo, fo_mask=None, chunk_size=1024, threshold=-30, nodata_out=-9999):
    """
    Fill gaps in fi using fi_fill only where esa == 80 and fi is nodata or below threshold.
    For gaps where esa != 80, encode as np.nan (nodata).
    Save output to fo, nodata set to nodata_out.
    Optionally save binary mask of filled pixels to fo_mask.
    
    Parameters:
    - fi: str, input raster path
    - fi_fill: str, fill raster path (must align with fi)
    - esa: str, raster path for ESA data (must align with fi)
    - fo: str, output raster path for filled data
    - fo_mask: str or None, output path for binary filled mask (1 = filled)
    - chunk_size: int, size of chunks/windows for processing
    - threshold: float, values below this in fi are considered invalid and filled
    - nodata_out: float/int, nodata value for output raster
    """
    ti = time.perf_counter()
    if os.path.isfile(fo):
        print(f"Output file {fo} already exists. Skipping processing.")
        return fo

    if fo_mask and os.path.isfile(fo_mask):
        print(f"Mask file {fo_mask} already exists. Skipping mask creation.")

    with rasterio.open(fi) as src, rasterio.open(fi_fill) as src_fill, rasterio.open(esa) as src_esa:
        profile = src.profile.copy()
        profile.update(nodata=nodata_out)
        nodata_in = src.nodata if src.nodata is not None else -9999

        # Mask profile (single band uint8, 0 or 1)
        mask_profile = profile.copy()
        mask_profile.update(dtype=rasterio.uint8, nodata=0, count=1)

        # Check alignment for all three rasters
        for r in [src_fill, src_esa]:
            if (src.width != r.width or 
                src.height != r.height or 
                src.transform != r.transform or
                src.crs != r.crs):
                raise ValueError("Input rasters do not match in shape, CRS, or transform.")

        height, width = src.height, src.width
        num_chunks_y = (height - 1) // chunk_size + 1
        num_chunks_x = (width - 1) // chunk_size + 1

        with rasterio.open(fo, 'w', **profile) as dst:
            mask_dst = None
            if fo_mask and not os.path.isfile(fo_mask):
                mask_dst = rasterio.open(fo_mask, 'w', **mask_profile)

            with tqdm(total=num_chunks_y * num_chunks_x, desc="Filling gaps with ESA mask") as pbar:
                for row in range(0, height, chunk_size):
                    for col in range(0, width, chunk_size):
                        win_width = min(chunk_size, width - col)
                        win_height = min(chunk_size, height - row)
                        window = Window(col, row, win_width, win_height)

                        data = src.read(1, window=window).astype(np.float32)
                        fill_data = src_fill.read(1, window=window).astype(np.float32)
                        esa_data = src_esa.read(1, window=window).astype(np.float32)

                        # Mark invalid pixels in fi
                        invalid_mask = (data == nodata_in) | (data < threshold)
                        data[invalid_mask] = np.nan

                        # Where esa == 80 and data is nan and fill_data valid and > threshold
                        fill_mask = (esa_data == 80) & np.isnan(data) & (~np.isnan(fill_data)) & (fill_data > threshold)
                        data[fill_mask] = fill_data[fill_mask]

                        # For remaining np.nan pixels (gaps) where esa != 80, keep as np.nan

                        # Replace np.nan with nodata_out before writing
                        data_to_write = np.where(np.isnan(data), nodata_out, data)

                        dst.write(data_to_write.astype(profile['dtype']), 1, window=window)

                        # Write mask if requested
                        if mask_dst:
                            mask_arr = np.zeros_like(data, dtype=np.uint8)
                            mask_arr[fill_mask] = 1
                            mask_dst.write(mask_arr, 1, window=window)

                        pbar.update(1)

            if mask_dst:
                mask_dst.close()

    tf = time.perf_counter() - ti
    print(f"\nFilled raster saved to: {fo}")
    if fo_mask:
        print(f"Binary filled mask saved to: {fo_mask}")
    print(f"Time taken: {tf:.2f} seconds")
    return fo

def gfill_with_constant(fi, fo, k=0, chunk_size=1024):
    ti = time.perf_counter()
    if os.path.isfile(fo):
        print(f"Output file {fo} already exists. Skipping processing.")
        return fo

    with rasterio.open(fi) as src:
        profile = src.profile.copy()
        nodata = src.nodata if src.nodata is not None else -9999
        profile.update(nodata=nodata)

        height, width = src.height, src.width

        with rasterio.open(fo, 'w', **profile) as dst:
            total_chunks = ((height - 1) // chunk_size + 1) * ((width - 1) // chunk_size + 1)

            for row in tqdm(range(0, height, chunk_size), desc="Processing rows"):
                for col in range(0, width, chunk_size):
                    win_width = min(chunk_size, width - col)
                    win_height = min(chunk_size, height - row)
                    window = Window(col, row, win_width, win_height)

                    data = src.read(1, window=window).astype(np.float32)

                    # Set nodata and values < -30 to np.nan
                    mask = (data == nodata) | (data < -30)
                    data[mask] = np.nan

                    # Replace np.nan with constant k
                    data = np.nan_to_num(data, nan=k)

                    dst.write(data, 1, window=window)

    tf = time.perf_counter() - ti
    print(f"\nProcessed raster saved to: {fo}")
    print(f"Time taken: {tf:.2f} seconds")
    return fo


def gdal_fillnodata(src_path, dst_path, md=100, si=0,method="inv_dist", output_format="GTiff", band=1):
    print(f'Infile ... {dst_path}')
    """
    Calls gdal_fillnodata.py via subprocess to interpolate nodata areas in a raster.

    Parameters:
    - src_path: str, input raster file path (with gaps to fill)
    - dst_path: str, output raster path
    - max_distance md: int, maximum search distance for interpolation (GDAL -md)
    - smoothing_iterations si: int, optional smoothing iterations (GDAL -si)
    - interp_method method: str, 'inv_dist' or 'nearest'
    - output_format: str, output GDAL format (default: GTiff)
    - band: int, which band to process (default: 1)

    Raises:
    - subprocess.CalledProcessError if GDAL call fails
    """
    if method not in ["inv_dist", "nearest"]:
        raise ValueError("Invalid interpolation method. Use 'inv_dist' or 'nearest'.")
    dst_path = dst_path.replace(".tif", f"_{si}_{md}_{method}.tif")
    if os.path.isfile(dst_path):
        print(f"Output file {dst_path} already exists. Skipping processing.")
        return dst_path
    
    gdal_cmd = [
        "gdal_fillnodata.py",
        "-md", str(md),
        "-si", str(si),
        "-of", output_format,
        "-b", str(band),
        "-interp", method,
        src_path,
        dst_path
    ]
    subprocess.run(gdal_cmd, check=True)
    print(dst_path)

def rio_fillnodata():
    # this is more expensive than gdal_fillnodata?
    #add this function into the data 
    pass 