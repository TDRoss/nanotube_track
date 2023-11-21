import os
import argparse
import cv2
import nd2reader
import pandas as pd
import numpy as np
from skimage import filters, measure, morphology, restoration, color
import trackpy as tp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def import_nd2(file_path):
    images = nd2reader.ND2Reader(file_path)
    return images

def segment_filaments(image): 
    background = filters.gaussian(image, sigma=1)
    subtracted_image = image - background
    subtracted_image_normalized = (subtracted_image - subtracted_image.min()) / (subtracted_image.max() - subtracted_image.min())
    local_thresh = filters.threshold_otsu(subtracted_image_normalized)
    segmented = subtracted_image_normalized > local_thresh
    return segmented

def remove_non_filaments(segmented_image):
    # Skeletonization
    skeleton = morphology.skeletonize(segmented_image)

    # Label the objects in the skeleton
    labeled_skeleton = measure.label(skeleton, connectivity=2)

    # # Iterate over each object and check for branch points
    for region in measure.regionprops(labeled_skeleton):
        # Identify a branch point by looking at the number of pixels
        # that have more than two neighbors in the skeleton
        if region.area <= 4: #minimum size threshold
            skeleton[labeled_skeleton == region.label] = 0
        else:
            coords = region.coords
            for y, x in coords:
                neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2]) - 1  # Subtract 1 for the pixel itself
                if neighbors > 2:
                    # If a branch point is found, remove this object
                    skeleton[labeled_skeleton == region.label] = 0
                    break
    return skeleton

def find_centroids(skeleton):
    labeled_skeleton = measure.label(skeleton)
    centroids = [props.centroid for props in measure.regionprops(labeled_skeleton)]
    return centroids

def track_filaments(image_sequence):
    # Use trackpy to track particles
    frames_data = []
    for frame_number, frame in enumerate(image_sequence): 
        centroids = find_centroids(frame) 
        y = [coord[0] for coord in centroids]
        x = [coord[1] for coord in centroids]
        fnum = [frame_number for _ in centroids]
        fdat = pd.DataFrame({'x': x, 'y': y, 'frame': fnum})
        frames_data.append(fdat)
    frames_data = pd.concat(frames_data)
    t = tp.link(frames_data, search_range=2) #search_range is # pixels to search for an object from one frame to the next
    t1 = tp.filter_stubs(t, 5) #second value is minimum number of frames for tracks that we keep
    return t1

def apply_drift_correction(tracks):
    drift = tp.compute_drift(tracks)
    corrected_tracks = tp.subtract_drift(tracks,drift)
    return corrected_tracks

def calculate_speed(tracks,frame_rate, micron_per_pixel):
    speeds = tracks.groupby('particle').apply(lambda x: micron_per_pixel * frame_rate * np.sqrt((np.diff(x['x'])**2 + np.diff(x['y'])**2).sum()) / (x['frame'].max() - x['frame'].min()))
    return speeds

#For overlaying centroid over masks


def overlay_centroids(cleaned_img, centroids):
    overlay_image = color.gray2rgb(cleaned_img)
    overlay_image = (overlay_image * 255).astype(np.uint8)
    for _, rows in centroids.iterrows():
        x = rows['x']
        y = rows['y']
        cv2.circle(overlay_image, (int(x), int(y)), 5, (255, 0, 0), -1)  # Red circle at each centroid
    return overlay_image


def run_tubetrack(options=None):
    frame_rate = 1/30 #frames/seconds
    micron_per_pixel = 1/6.25
    # Calculate the figure size in inches (512 pixels / 300 DPI)
    fig_size = 512 / 300 * 2  # 2 is a scaling factor

    if options is None or 'file' not in options:
        file_path = input("Please provide the path to the file to be analyzed.\n")
    else:
        file_path = options["file"]
    images = import_nd2(file_path)
    segmented_images = [segment_filaments(image) for image in images]
    cleaned_images = [remove_non_filaments(image) for image in segmented_images]
    tracks = track_filaments(cleaned_images)
    corrected_tracks = apply_drift_correction(tracks)

    frames = []
    for i in range(len(cleaned_images)):
        current_frame_tracks = corrected_tracks[corrected_tracks['frame'] == i]
        centroids = current_frame_tracks[['x', 'y']]
        overlay_image = overlay_centroids(cleaned_images[i], centroids)
        frames.append(overlay_image)

    filename, _ = os.path.splitext(file_path)
    # Create and save the movie
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=300)    
    image_plot = ax.imshow(frames[0], animated=True)
    plt.axis('off')

    def update_frame(i):
        image_plot.set_data(frames[i])
        return [image_plot]

    animation = FuncAnimation(fig, update_frame, frames=len(frames), blit=True)
    animation.save(f'{filename}_tracking.mp4', writer='ffmpeg')


    speeds = calculate_speed(corrected_tracks, frame_rate, micron_per_pixel)
    # Save speed histogram
    fig = plt.figure(figsize=(fig_size, fig_size), dpi=300)
    plt.hist(speeds*1000, bins=30, alpha=0.75, color='blue', edgecolor='black')
    plt.xlabel('Speed (nm/s)')
    plt.ylabel('Counts')
    plt.savefig(f'{filename}_speed_histogram.png')


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description="Analyze nanotube microscopy data")
        parser.add_argument("--file", type=str, help="Path to nd2 file")
        args = parser.parse_args()
        options = {}
        if args.file:
            options['file'] = args.file
    except:
        options = None
    run_tubetrack(options)