import itertools
import json
import pandas as pd
import numpy as np
from dotenv import dotenv_values
import glob
import os
import tqdm

from czifile import imread  # for reading CZI files.
from skimage import io  # for reading TIFF files.
import tifffile
import scipy
from scipy.stats import multivariate_normal
from scipy.signal import find_peaks, peak_widths
from itertools import product

DATABASE_FILE_PATHS = []
DATA_OUTPUT_PATH = '.'
CENTROMERE_CHANEL = None
BULK_CHANNEL = None
VOXEL_SIZE = [0.0313, 0.0313, 0.2]
JSON_KEYS = ['cell_id', 'spindle_pole_track_path', 'centromere_track_path', 'image_path', 'strain', 'Centromere_path']#, 'Kinetochore_path']


class DatabaseExtractionException(Exception):
    pass


def read_czi(path) -> np.ndarray:
    """
    Read a czi file and return the resulting image file as a numpy array.

    Results will not be normalized as they are "scaled to raw" in the SIM processing so they can be compared to
    each other.

    The indexing order for this array is [CHANNEL, FRAME, X, Y, Z].

    :param path: The path to the .czi file.
    :return: The `numpy.ndarray` containing the data. The index order is `[C, F, X, Y, Z]`.
    """
    # This will return a numpy array with different data indexed by
    # [?,?,channel,frame,z,x,y,?]
    image = imread(path)
    image = image[0, 0, :, :, :, :, :, 0].astype(np.float64)
    # image = image / np.max(image)
    # Image is now indexed by [channel, frame, z, y, x] and normalized.

    # Reshuffle indices to be [channel, frame, x, y, z]
    image = np.moveaxis(image, [0, 1, 2, 3, 4], [1, 0, 4, 3, 2])

    return image


def read_tiff(path) -> np.ndarray:
    """
    Read a tif/tiff file and return the resulting image file as a numpy array.

    Results will not be normalized as they are "scaled to raw" in the SIM processing so they can be compared to
    each other.

    The indexing order for this array is [CHANNEL, FRAME, X, Y, Z].

    :param path: The path to the .tif file.
    :return: The `numpy.ndarray` containing the data. The index order is `[F, C, X, Y, Z]`.
    """
    image = tifffile.imread(path)
    with tifffile.TiffFile(path) as tif:
        axes_order = tif.series[0].axes
        t = axes_order.index('T')
        c = axes_order.index('C')
        x = axes_order.index('X')
        y = axes_order.index('Y')
        z = axes_order.index('Z')

        image = np.transpose(image, (t, c, x, y, z))  # Reorder the image to be indexed correctly.

    return image


def nth_largest_value(dictionary, n):
    """
    Return the key corresponding ot the nth largest value in the dictionary.

    :param dictionary: The dictionary to search.
    :param n: The nth largest value to search for (0=largest, 1=second largest, ...)
    :return: The key corresponding to the nth largest value.
    """
    values_list = list(dictionary.values())

    values_list.sort(reverse=True)

    if n <= len(values_list):
        nlv = values_list[n]

        for key, value in dictionary.items():
            if value == nlv:
                return key

    return None  # Return None if n is out of range


def identify_new_pole_track_id(track_df: pd.DataFrame, metric="MEDIAN_INTENSITY_CH1") -> int:
    """
    A function to identify the TRACK_ID for the new spindle pole in the resulting trackmate csv file.

    It will achieve this by looking at the track which has spots with the second-highest metric on average.

    (This metric can and should be changed depending on the tracking methods used.)

    :param track_df: The pandas DataFrame containing the spindle pole tracking data.
    :param metric: The column metric to use to assess the old vs new spindle.
    :return: The TRACK_ID for the old spindle pole in the given track results.
    """
    t_ids = np.unique(track_df['TRACK_ID'])
    if len(t_ids) == 2:
        return t_ids[0]

    track_ids = np.unique(track_df['TRACK_ID'])
    average_qualities = {}

    for i in track_ids:
        tdf = track_df[track_df['TRACK_ID'] == i]  # Subset the df to only contain this track.
        average_qualities[i] = np.mean(tdf[metric])

    return nth_largest_value(average_qualities, 1)


def identify_old_pole_track_id(track_df: pd.DataFrame, metric="MEDIAN_INTENSITY_CH1") -> int:
    """
    A function to identify the TRACK_ID for the old spindle pole in the resulting trackmate csv file.

    It will achieve this by looking at the track which has spots with the highest metric on average.

    (This metric can and should be changed depending on the tracking methods used.)

    :param track_df: The pandas DataFrame containing the spindle pole tracking data.
    :param metric: The column metric to use to assess the old vs new spindle.
    :return: The TRACK_ID for the old spindle pole in the given track results.
    """
    t_ids = np.unique(track_df['TRACK_ID'])
    if len(t_ids) == 2:
        return t_ids[1]

    track_ids = np.unique(track_df['TRACK_ID'])
    average_qualities = {}

    for i in track_ids:
        tdf = track_df[track_df['TRACK_ID'] == i]  # Subset the df to only contain this track.
        average_qualities[i] = np.mean(tdf[metric])

    return nth_largest_value(average_qualities, 0)


def identify_centromere_track_id(track_df: pd.DataFrame):
    return scipy.stats.mode(track_df["TRACK_ID"]).mode


def get_voxel_size(path):
    """
    Implemented based on information found in https://pypi.org/project/tifffile
    Found on: https://forum.image.sc/t/reading-pixel-size-from-image-file-with-python/74798/2
    """

    def _xy_voxel_size(tags, key):
        assert key in ['XResolution', 'YResolution']
        if key in tags:
            num_pixels, units = tags[key].value
            return units / num_pixels
        # return default
        return None

    with tifffile.TiffFile(path) as tiff:
        image_metadata = tiff.imagej_metadata
        if image_metadata is not None:
            z = image_metadata.get('spacing', 1.)
        else:
            # default voxel size
            z = None

        tags = tiff.pages[0].tags
        # parse X, Y resolution
        y = _xy_voxel_size(tags, 'YResolution')
        x = _xy_voxel_size(tags, 'XResolution')
        # return voxel size

        voxel_size = np.array([x, y, z])
        return voxel_size


def read_image_file(path) -> np.ndarray:
    """
    Read a czi file and return the resulting image file as a numpy array.

    Results will not be normalized as they are "scaled to raw" in the SIM processing so they can be compared to
    each other.

    The indexing order for this array is [FRAME, CHANNEL, X, Y, Z].

    :param path: The path to the .czi file.
    :return: The `numpy.ndarray` containing the data. The index order is `[F, C, X, Y, Z]`.
    """
    filename, file_extension = os.path.splitext(path)
    if file_extension == '.czi':
        return read_czi(path)
    elif file_extension == '.tif' or file_extension == '.tiff':
        return read_tiff(path)
    else:
        raise ValueError(f"{path} is not a recognised file type.")


def load_trackmate_data(path) -> pd.DataFrame:
    """
    Load data from a trackmate csv output. This function reads the file, removes unnecessary headers,
    and converts datatypes to numeric types before returning the resulting dataframe.

    :param path: The path to the trackmate file to load data from.
    :return: A pandas DataFrame containing the data from the trackmate csv output (properly formatted.)
    """
    try:
        res = pd.read_csv(path).iloc[3:].reset_index(drop=True).drop(columns='LABEL')
    except Exception as e:
        try:
            res = pd.read_csv(path).iloc[3:].reset_index(drop=True).drop(columns='Label')
        except Exception as e:
            res = pd.read_csv(path).iloc[3:].reset_index(drop=True)
    res = res.apply(pd.to_numeric)

    return res


def compute_spindle_length(df, frame, op_track_id, np_track_id):
    tdf = df[df['FRAME'] == frame]
    op_df = tdf[tdf['TRACK_ID'] == op_track_id]
    np_df = tdf[tdf['TRACK_ID'] == np_track_id]

    if (len(op_df.index) != 1) or (len(np_df.index) != 1):
        return np.nan

    op_pos = np.array([
        op_df["POSITION_X"].iloc[0],
        op_df["POSITION_Y"].iloc[0],
        op_df["POSITION_Z"].iloc[0]
    ])
    np_pos = np.array([
        np_df["POSITION_X"].iloc[0],
        np_df["POSITION_Y"].iloc[0],
        np_df["POSITION_Z"].iloc[0]
    ])

    return np.sqrt(np.sum(np.square(op_pos - np_pos)))


def get_foci_points(df, frame):
    track_id = identify_centromere_track_id(df)
    tdf = df[df['FRAME'] == frame]
    tdf = tdf[tdf['TRACK_ID'] == track_id]

    if len(tdf.index) == 0:
        return []
    elif len(tdf.index) == 1:
        pos = np.array([
            tdf["POSITION_X"].iloc[0],
            tdf["POSITION_Y"].iloc[0],
            tdf["POSITION_Z"].iloc[0]
        ])
        return [pos]
    elif len(tdf.index) == 2:
        pos_one = np.array([
            tdf["POSITION_X"].iloc[0],
            tdf["POSITION_Y"].iloc[0],
            tdf["POSITION_Z"].iloc[0]
        ])
        pos_two = np.array([
            tdf["POSITION_X"].iloc[1],
            tdf["POSITION_Y"].iloc[1],
            tdf["POSITION_Z"].iloc[1]
        ])
        return [pos_one, pos_two]
    else:
        raise ValueError("Unexpected number of centromere/bulk foci.")


def get_kinetichore_mt_distance(bulk_df, spindle_df, frame, op_track_id, np_track_id) -> float:
    # Get the spindle pole positions.
    tdf = spindle_df[spindle_df['FRAME'] == frame]
    op_df = tdf[tdf['TRACK_ID'] == op_track_id]
    np_df = tdf[tdf['TRACK_ID'] == np_track_id]

    if (len(op_df.index) != 1) or (len(np_df.index) != 1):
        return np.nan

    op_pos = np.array([
        op_df["POSITION_X"].iloc[0],
        op_df["POSITION_Y"].iloc[0],
        op_df["POSITION_Z"].iloc[0]
    ])
    np_pos = np.array([
        np_df["POSITION_X"].iloc[0],
        np_df["POSITION_Y"].iloc[0],
        np_df["POSITION_Z"].iloc[0]
    ])

    spindle_points = [op_pos, np_pos]

    # Get the bulk points.
    bulk_points = get_foci_points(bulk_df, frame)

    distances = [np.sqrt(np.sum(np.square(spindle_point - bulk_point))) for spindle_point, bulk_point in product(spindle_points, bulk_points)]
    return np.min(distances) if len(distances) > 0 else np.nan

def compute_centromere_separation_from_points(points):
    if len(points) == 2:
        pos_one, pos_two = points
        return np.sqrt(np.sum(np.square(pos_one - pos_two)))
    else:
        return np.nan

def gauss(x, mu, sigma, A):
    return A * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2)

def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gauss(x, mu1, sigma1, A1) + gauss(x, mu2, sigma2, A2)

def _initial_guesses(x, y, bounds, n_restarts=8, seed=0):
    """Seeds for the local fit: peak-detection guesses plus random restarts."""
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    dx = np.median(np.diff(x))

    guesses = []
    peaks, props = find_peaks(y, prominence=0.05 * np.max(y))
    if len(peaks):
        peaks = peaks[np.argsort(props['prominences'])[::-1]]
        widths = peak_widths(y, peaks, rel_height=0.5)[0] * dx / 2.3548
        widths = np.clip(widths, lo[1], hi[1])
        if len(peaks) >= 2:
            i, j = peaks[0], peaks[1]
            guesses.append([x[i], widths[0], y[i], x[j], widths[1], y[j]])
        i = peaks[0]  # single peak: seed as two overlapping components
        guesses.append([x[i] - widths[0], widths[0], y[i],
                        x[i] + widths[0], widths[0], y[i]])

    rng = np.random.default_rng(seed)  # fixed seed keeps fits reproducible
    guesses += [lo + rng.random(6) * (hi - lo) for _ in range(n_restarts)]
    return [np.clip(np.asarray(g, float), lo + 1e-12, hi - 1e-12) for g in guesses]


def err_fit(x, y, bounds):
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    best_params, best_err = None, np.inf
    for p0 in _initial_guesses(x, y, bounds):
        try:
            params, _ = scipy.optimize.curve_fit(
                bimodal, x, y, p0=p0, bounds=(lo, hi), maxfev=20000
            )
        except (RuntimeError, ValueError):
            continue
        err = np.mean(np.square(y - bimodal(x, *params)))
        if err < best_err:
            best_err, best_params = err, params

    if best_params is None:  # every restart failed - fall back to the global search
        return scipy.optimize.dual_annealing(
            lambda p: np.mean(np.square(y - bimodal(x, *p))), bounds
        ).x
    return best_params

def fit_bimodal_gaussian(data: pd.DataFrame, normalize: bool = True):
    """
    Fit a bimodal Gaussian (two-component GMM) to histogram data and return the fit parameters.

    :param data: A DataFrame where the first column is distance and the second is intensity.
    :param normalize: If True, normalize y by its max before fitting (default). If False, fit on raw y values.
    :return: A dict with keys:
        - 'x': the distance array
        - 'y': the intensity array used for fitting (normalized if normalize=True, raw otherwise)
        - 'params': tuple of (mu1, sigma1, A1, mu2, sigma2, A2)
        - 'distance': the absolute distance between the two peaks
    """
    x = np.array(data[data.columns[0]])
    y = np.array(data[data.columns[1]])
    y = y - np.min(y)  # Shift y to be non-negative for fitting.

    if normalize:
        y = y / np.max(y)

    max_sigma = 0.3
    min_sigma = 0.01

    if normalize:
        min_intensity = 0.2
        max_intensity = 1.0
    else:
        min_intensity = 0.2 * np.max(y)
        max_intensity = np.max(y)

    bounds_buffer = 0.2 * np.max(x)
    bounds = [
        (0 - bounds_buffer, np.max(x) + bounds_buffer),
        (min_sigma, max_sigma),
        (min_intensity, max_intensity),
        (0 - bounds_buffer, np.max(x) + bounds_buffer),
        (min_sigma, max_sigma),
        (min_intensity, max_intensity)
    ]

    params = err_fit(x, y, bounds)
    dist = np.abs(params[0] - params[3])

    return {
        'x': x,
        'y': y,
        'params': tuple(params),
        'distance': dist
    }


def compute_centromere_separation_gaussian(data: pd.DataFrame, normalize: bool = True) -> float:
    result = fit_bimodal_gaussian(data, normalize=normalize)
    return result['distance']


def load_json(path):
    with open(path, 'r') as f:
        j = json.load(f)
    for k in JSON_KEYS:  # Check that all keys are present for the database.
        if k not in j.keys() or j[k] is None:
            raise DatabaseExtractionException(f"Key {k} not present in database entry: {path}")
    return j


def get_dir(path):
    return os.path.dirname(path)


def process_database_cell(path) -> pd.DataFrame:
    # Load the JSON for the database entry.
    cell_data = load_json(path)
    if 'Kinetochore_track_path' not in cell_data.keys():
        cell_data['Kinetochore_track_path'] = ''

    # Load the individual histogram data for the centromere separation if it exists.
    centromere_histogram_path = os.path.join(os.path.dirname(path), cell_data['Centromere_path'])
    bulk_histogram_path = None#os.path.join(os.path.dirname(path), cell_data['Kinetochore_path'])
    if not os.path.exists(centromere_histogram_path):
        centromere_histogram_path = None
    # if not os.path.exists(bulk_histogram_path):
    #     bulk_histogram_path = None

    # Load the data we will synthesize to get the output.
    image_path = os.path.join(get_dir(path), cell_data['image_path'])
    image_data = read_image_file(image_path)
    spindle_pole_track_path = os.path.join(get_dir(path), cell_data['spindle_pole_track_path'])
    spindle_pole_track_data = load_trackmate_data(spindle_pole_track_path)
    centromere_track_path = os.path.join(get_dir(path), cell_data['centromere_track_path']) if cell_data['centromere_track_path'] != "" else None
    centromere_track_data = load_trackmate_data(centromere_track_path) if centromere_track_path is not None else None
    bulk_track_path = os.path.join(get_dir(path), cell_data['Kinetochore_track_path']) if cell_data['Kinetochore_track_path'] != "" else None # TODO: What is the key for this?
    bulk_track_data = load_trackmate_data(bulk_track_path) if bulk_track_path is not None else None

    # Iterate over the frames of the video in order to get the data.
    if centromere_track_data is not None:
        num_frames = max(image_data.shape[0],
                         max(np.max(spindle_pole_track_data['FRAME']), np.max(centromere_track_data['FRAME'])))
    else:
        num_frames = max(image_data.shape[0], np.max(spindle_pole_track_data['FRAME']))
    spindle_length = []
    centromere_separation_trackmate = []
    centromere_separation_gaussian = []
    bulk_separation_gaussian = []
    estimated_kinetichore_mt_distance = []
    frames = np.array(range(1, num_frames + 1))
    op_track_id = identify_old_pole_track_id(spindle_pole_track_data)
    np_track_id = identify_new_pole_track_id(spindle_pole_track_data)
    for f in frames:
        spindle_length.append(compute_spindle_length(spindle_pole_track_data, f, op_track_id, np_track_id))
        if centromere_track_data is not None:
            centromere_points = get_foci_points(centromere_track_data, f)
            centromere_separation_trackmate.append(
                compute_centromere_separation_from_points(centromere_points)
            )
        else:
            centromere_separation_trackmate.append(np.nan)
        if centromere_histogram_path is not None:
            try:
                p = os.path.join(centromere_histogram_path, f'{f-1}.csv')
                centromere_separation_gaussian.append(
                    compute_centromere_separation_gaussian(pd.read_csv(p))
                )
            except FileNotFoundError as e:
                centromere_separation_gaussian.append(np.nan)
        else:
            centromere_separation_gaussian.append(np.nan)
        if bulk_histogram_path is not None:
            try:
                p = os.path.join(bulk_histogram_path, f'{f-1}.csv')
                bulk_separation_gaussian.append(
                    compute_centromere_separation_gaussian(pd.read_csv(p))
                )
            except FileNotFoundError as e:
                bulk_separation_gaussian.append(np.nan)
        else:
            bulk_separation_gaussian.append(np.nan)
        if bulk_track_data is not None:
            estimated_kinetichore_mt_distance.append(get_kinetichore_mt_distance(bulk_track_data, spindle_pole_track_data, f, op_track_id, np_track_id))
        else:
            estimated_kinetichore_mt_distance.append(np.nan)

    res = pd.DataFrame({
        'Spindle Length': spindle_length,
        'Centromere Separation': centromere_separation_gaussian,
        'Bulk Separation': bulk_separation_gaussian,
        'Estimated kMT Length': estimated_kinetichore_mt_distance,
    },
        index=frames
    )

    return res


if __name__ == '__main__':
    config = dotenv_values('./database_processing_settings.env')
    DATABASE_FILE_PATHS = np.array(glob.glob(config['INPUT_PATH'], recursive=True))
    DATA_OUTPUT_PATH = config['OUTPUT_PATH']
    VOXEL_SIZE = np.array([float(t) for t in config['VOXEL_SIZE'][1:-1].split(",")])
    print("Loaded environment data.")
    print("\tINPUT_PATH:", config['INPUT_PATH'])
    print("\tOUTPUT_PATH:", config['OUTPUT_PATH'])
    print("\tVOXEL_SIZE:", config['VOXEL_SIZE'])
    print(f"Processing {len(DATABASE_FILE_PATHS)} files")
    print(DATABASE_FILE_PATHS)

    processed_dataframes = {}
    for cell_entry_path in tqdm.tqdm(DATABASE_FILE_PATHS):
        try:
            df = process_database_cell(cell_entry_path)
            cell_id = load_json(cell_entry_path)['cell_id']
            strain = load_json(cell_entry_path)['strain']

            output_dir = os.path.join(DATA_OUTPUT_PATH, strain)
            os.makedirs(output_dir, exist_ok=True)
            df.to_csv(os.path.join(output_dir, f"{cell_id}.csv"))
        except DatabaseExtractionException as e:
            print("Error reading database json file:", cell_entry_path)
            continue
        except Exception as e:
            print(f"Error Processing file at: {cell_entry_path}")
            print(e, '\n')
            continue
            # raise e
