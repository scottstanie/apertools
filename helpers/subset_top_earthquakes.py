from datetime import date
import numpy as np
import apertools.sario as sario
import apertools.subset as subset
import pandas as pd

# from insar.prepare import remove_ramp
# from insar.timeseries import PHASE_TO_CM


def read_eqs(fname="../../fracking-qgis-data/texnet_events_20200726.csv"):
    df = pd.read_csv(
        fname,
        usecols=(
            "Origin Date",
            "Origin Time",
            "Latitude",
            "Longitude",
            "Magnitude",
            "Depth of Hypocenter (Km.  Rel to MSL)",
        ),
        parse_dates=[["Origin Date", "Origin Time"]],
    )
    df.columns = ["dt", "mag", "lat", "lon", "depth"]
    return df.set_index("dt")


def get_top_mag_dates(df, n=None):
    dailydf = df.copy()
    # dailydf.reset_index(drop=True, inplace=True)  # turns into RangeIndex
    dailydf["date"] = dailydf.index.date
    dailydf = dailydf.groupby("date")["mag"].agg(max_mag="max", idxmax="idxmax")
    dailydf[["lat", "lon"]] = df.loc[dailydf["idxmax"], ["lat", "lon"]].values
    dailydf = (
        dailydf[["max_mag", "lat", "lon"]]
        .sort_values("max_mag", ascending=False)
        .head(n)
    )
    dailydf["bbox"] = subset.bbox_around_point(dailydf["lon"], failydf["lat"])
    return dailydf
