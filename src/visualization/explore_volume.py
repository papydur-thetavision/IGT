from src.io.volume_reader import VolumeReader
import plotly.express as px
import plotly.graph_objects as go
import napari
import numpy as np


def plot_2d(volume, mask):
    img = volume[60, :, :]
    mask = mask[60, :, :]

    fig = px.imshow(img, color_continuous_scale='gray')
    fig.add_trace(go.Contour(z=mask, ncontours=2, showscale=False, contours=dict(coloring='lines'), line=dict(width=2)))

    fig.update_layout(coloraxis=dict(colorscale='gray'), showlegend=False)
    fig.show()


def show_3d_volume(volume_reader, index=0):

    volume, mask = volume_reader[index]
    skeleton = volume_reader.skeletonize_mask(mask)
    viewer = napari.Viewer(ndisplay=3)

    viewer.add_image(volume, rgb=False)
    viewer.add_image(mask, rgb=False, blending='additive', colormap='cyan')

    points = volume_reader.get_end_point_from_skeleton(skeleton)
    viewer.add_points(points, size=5)

    napari.run()


def plot_from_prediction(volume, y, y_pred):
    volume = np.array(volume.squeeze().detach())
    volume_shape = volume.shape
    viewer = napari.Viewer(ndisplay=3)

    viewer.add_image(volume, rgb=False)

    points = [
        np.array([int(axis*volume_shape[0]) for axis in y['coord1']]),
        np.array([int(axis*volume_shape[0]) for axis in y['coord2']]),
    ]

    points_pred = [
        np.array([int(axis*volume_shape[0]) for axis in y_pred['coord1']]),
        np.array([int(axis*volume_shape[0]) for axis in y_pred['coord2']]),
    ]

    viewer.add_points(points, size=5, edge_color='green')
    viewer.add_points(points_pred, size=5, edge_color='blue')

    napari.run()


if __name__ == '__main__':

    vr = VolumeReader()
    vr.create_volume_list()
    show_3d_volume(vr, index=2)
