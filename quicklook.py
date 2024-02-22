import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st

parser = argparse.ArgumentParser()
parser.add_argument("db")

SELECT_KEYS = {
    "model_amp",
    "model_fwhm",
    "max",
    "median",
    "mean",
    "sum",
    "var",
    "normvar",
    "photflux",
}
CENTROIDS = {"com", "peak", "model"}
CENTROID_KEYS = {"com_x", "com_y", "peak_x", "peak_y", "model_x", "model_y"}


def main():
    args = parser.parse_args()
    db = pd.read_csv(args.db, index_col=0)

    pnames = [str(Path(p).name) for p in db["path"]]
    curr_file = st.selectbox("Filename", pnames)

    idx = pnames.index(curr_file)

    metric_file = db["metric_file"].iloc[idx]

    metrics = np.load(metric_file)
    key_set = set(metrics.keys())

    st.header("Frame Statistics")
    average = st.toggle("Average values", value=True)
    select_keys = list(sorted(SELECT_KEYS & key_set))
    tabs = st.tabs(select_keys)
    for key, tab in zip(select_keys, tabs, strict=True):
        fig = go.Figure()
        mean_metric = metrics[key].mean(0)
        if average:
            trace = go.Scatter(dict(y=mean_metric), mode="markers", name="Average")
            fig.add_trace(trace)
        else:
            for i, met in enumerate(metrics[key]):
                trace = go.Scatter(dict(y=met), mode="markers", name=f"PSF {i}")
                fig.add_trace(trace)

        fig.update_layout(
            legend_orientation="h",
            legend_yanchor="bottom",
            legend_y=1.02,
            legend_xanchor="left",
            legend_x=0.0,
        )
        tab.plotly_chart(fig)
        # trace = go.Scatter(metrics[key])
        # tab.line_chart(metrics[key].T)

    st.subheader("Centroids")
    methods = list(sorted(filter(lambda k: any(s.startswith(k) for s in key_set), CENTROIDS)))
    tabs = st.tabs(methods)
    for key, tab in zip(methods, tabs, strict=True):
        xs = metrics[f"{key}_x"]
        ys = metrics[f"{key}_y"]
        if average:
            fig = ff.create_2d_density(xs.mean(0), ys.mean(0))
            # trace = go.Scatter(dict(x=xs.mean(0), y=ys.mean(0)), mode="markers", name="Average")
            # fig.add_trace(trace)
        else:
            fig = go.Figure()
            for i, (px, py) in enumerate(zip(xs, ys, strict=True)):
                trace = go.Scatter(dict(x=px, y=py), mode="markers", name=f"PSF {i}")
                fig.add_trace(trace)
        fig.update_layout(
            legend_orientation="h",
            legend_yanchor="bottom",
            legend_y=1.02,
            legend_xanchor="left",
            legend_x=0.0,
            width=600,
            height=600,
        )
        tab.plotly_chart(fig)


if __name__ == "__main__":
    main()
