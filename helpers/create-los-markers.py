from typing import Literal, Optional
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from pathlib import Path

from apertools.plotting import create_marker_from_svg


def create_los_marker(
    incidence_angle: float,
    direction: Literal["ascending", "descending"],
    looking: Literal["right", "left"] = "right",
    cmap: str = "RdBu_r",
    size: float = 1.5,
    output_pdf: Optional[str] = None,
) -> plt.Figure:
    """Create a Line of Sight (LOS) marker showing incidence angle with color gradient.

    Parameters
    ----------
    incidence_angle : float
        Angle in degrees from vertical
    direction : Literal["ascending", "descending"]
        Satellite orbit direction
    looking : Literal["right", "left"]
        Side the satellite is looking towards.
        Defaults to "right"
    cmap : str, optional
        Matplotlib colormap name, by default 'RdBu_r'
    size : float, optional
        Scaling factor for marker size, by default 1.0
    output_pdf : str, optional
        If provided, save figure to this PDF path

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the LOS marker
    """
    fig = plt.figure(figsize=(2 * size, 2 * size))
    ax = fig.add_subplot(111)

    # Draw compass points with full words
    compass_points = {
        "West": (-1.3, 0),
        "East": (1.3, 0),
        "Up": (0, 1.2),
        "Down": (0, -1.2),
    }
    for label, (x, y) in compass_points.items():
        ax.text(x, y, label, ha="center", va="center", fontsize=12)
    circle = Circle((0, 0), 1, fill=False, color="black", linewidth=1.5)
    ax.add_patch(circle, zorder=5)

    # Add light cross grid
    ax.plot([-1, 1], [0, 0], "lightgray", linewidth=0.5)
    ax.plot([0, 0], [-1, 1], "lightgray", linewidth=0.5)

    # Calculate arrow endpoint based on incidence angle
    angle_rad = np.radians(
        _get_incidence_bar_angle(incidence_angle, direction=direction, looking=looking)
    )
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)

    # Create gradient bar
    bar_width = 0.15
    bar_length = 2

    # Create points for gradient bar
    theta = np.arctan2(dy, dx)
    rot_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

    # Base rectangle points for gradient bar
    bar_points = np.array(
        [
            [-bar_length / 2, -bar_width / 2],
            [bar_length / 2, -bar_width / 2],
            [bar_length / 2, bar_width / 2],
            [-bar_length / 2, bar_width / 2],
        ]
    )

    cm = plt.get_cmap(cmap)
    # Create smooth gradient
    num_segments = 150  # More segments for smoother gradient
    for i in range(num_segments):
        t_start = i / num_segments
        t_end = (i + 1) / num_segments

        # Calculate points for this segment
        x_start = bar_points[0, 0] + (bar_points[1, 0] - bar_points[0, 0]) * t_start
        x_end = bar_points[0, 0] + (bar_points[1, 0] - bar_points[0, 0]) * t_end

        segment_points = np.array(
            [
                [x_start, -bar_width / 2],
                [x_end, -bar_width / 2],
                [x_end, bar_width / 2],
                [x_start, bar_width / 2],
            ]
        )

        # Rotate segment points
        rotated_segment = np.dot(segment_points, rot_matrix.T)

        # Plot segment with gradient color
        color = cm(1 - t_start)
        ax.fill(
            rotated_segment[:, 0],
            rotated_segment[:, 1],
            color=color,
            edgecolor=None,
            zorder=2,
        )

    # Plot the satellite icon above the circle.
    #  Need to flip/rotate so it's pointing the right way
    flip = direction == "ascending"
    sat_icon = create_marker_from_svg(
        Path(__file__).parent / "data/satellite.svg",
        rotation=180 if direction == "descending" else 0,
        flip=flip,
    )
    # Position is R cos(theta), R sin(theta), where R is 1.3
    R = -1.3
    sat_x = R * dx
    sat_y = R * dy
    ax.plot(sat_x, sat_y, marker=sat_icon, ms=40, color="black")

    # Set equal aspect ratio and limits
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis("off")

    if output_pdf:
        fig.savefig(output_pdf, bbox_inches="tight", dpi=300)

    return fig


def _get_incidence_bar_angle(
    incidence_angle: float,
    direction: Literal["ascending", "descending"],
    looking: Literal["right", "left"] = "right",
) -> float:
    """Calculate the angle for the incidence bar based on orbital parameters.

    Parameters
    ----------
    incidence_angle : float
        The incidence angle in degrees from vertical
    direction : Literal["ascending", "descending"]
        Satellite orbit direction
    looking : Literal["right", "left"]
        Side the satellite is looking towards, defaults to "right"

    Returns
    -------
    float
        Angle in degrees to use for the incidence bar orientation

    Notes
    -----
    - Ascending + right-looking: down & right (original -90° based system)
    - Ascending + left-looking: down & left
    - Descending + right-looking: down & left (as in second example)
    - Descending + left-looking: down & right
    """
    # Base angle from incidence angle
    base_angle = incidence_angle

    # Adjust for orbit direction
    if direction == "ascending":
        if looking == "right":
            # Original case: down and to the right
            return base_angle - 90
        else:  # looking left
            # Mirror across vertical
            return -base_angle - 90
    else:  # descending
        if looking == "right":
            # Down and to the left (as in second example)
            return -base_angle - 90
        else:  # looking left
            # Mirror of descending right-looking
            return base_angle - 90
