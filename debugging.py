import fastplotlib as fpl
from pathlib import Path
import dask.array as da
from dask import delayed
from mesmerize_viz import *
import mesmerize_core as mc


def get_df():
    parent_path = Path().home() / 'caiman_data' / 'raw' / 'mcorr.pickle'
    og_path = Path().home() / 'caiman_data' / 'raw' / 'roi_mc.pickle'
    df_og = mc.load_batch(og_path)
    df = mc.load_batch(parent_path)
    return df, df_og


def get_image_widget(df_idx=1):
    df, df_og = get_df()
    input_movie = df_og.iloc[df_idx].caiman.get_input_movie()
    corrected_og = df_og.iloc[df_idx].mcorr.get_output()

    lazy_arrays = []
    for i, row in df.iterrows():
        mov_delayed = delayed(row.mcorr.get_output)()
        mov_dask = da.from_delayed(
            mov_delayed,
            shape=(1729, 600, 145),
            dtype='float32'
        )
        lazy_arrays.append(mov_dask)
    result = da.concatenate(lazy_arrays, axis=-1)
    figure = fpl.ImageWidget(
        data=[input_movie, corrected_og, result.compute()],
        names=["Raw", "Registered (Post-Tiling)", "Registered (Pre-Tiling)"],
        histogram_widget=False,
        figure_kwargs={"size": (700, 560), "shape": (1, 3)},
    )
    for subplot in figure.figure:
        # sometimes the toolbar adds clutter
        subplot.toolbar = False
    return figure
    # return mviz_corr


def time_store():
    from mesmerize_viz._store_model import TimeStore
    return TimeStore()


def get_cnmf_viz():
    _, df_og = get_df()
    viz = df_og.cnmf.viz(start_index=10)
    return viz


if __name__ == '__main__':
    fig = get_image_widget(-2)
    fig2 = get_image_widget(-2)

    store = time_store()
    store.subscribe(fig, data=fig.data)
    store.subscribe(fig2, data=fig2.data)
    fig.show()
    fig2.show()
    fpl.run()

    # figure = get_image_widget()
    # viz = get_cnmf_viz()
    # viz.show()
    # figure.show()
    # fpl.run()
