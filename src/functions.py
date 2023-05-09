"""
Functions for analysis
"""

import itertools
from typing import List

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from IPython.core.display import display
from sklearn import decomposition

from src.constants import (
    ANALYZE_TARGET_COLS,
    CHUNK_SIZE,
    INTERMEDIATE_DATA_DIR,
    INTERMEDIATE_DATA_RATES,
    NOT_USED_COLUMNS,
    RAW_DATA_DIR,
    RAW_DATA_FILE,
    START_YEAR,
)


def chunk_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess with chunk load.

    Args:
        df (pd.DataFrame): Chunked data frame.

    Returns:
        pd.DataFrame: Preprocessed data frame.
    """

    # Drop records that have missing value
    df = df.dropna()

    # Singles data only
    df = df[df["doubles"] == "f"]

    # Target years
    df = df[df["year"] >= START_YEAR]

    # Drop not used columns
    df = df.drop(NOT_USED_COLUMNS, axis=1)

    # Only Grandslam
    df = df[df["masters"] == 2000]

    return df


def read_raw_data() -> pd.DataFrame:
    """Read raw data.

    Returns:
        pd.DataFrame: Read data.
    """

    # Read data using chunk processing
    reader = pd.read_csv(RAW_DATA_DIR / RAW_DATA_FILE, chunksize=CHUNK_SIZE)
    df_matches = pd.concat(
        (chunk_preprocess(r) for r in reader),
        ignore_index=True,
    )
    return df_matches


def save_intermediate_data(df: pd.DataFrame, save_path: str, index: bool) -> str:
    """Save data as an intermediate.

    Args:
        df (pd.DataFrame): Save data.
        save_path (str): Save path.
        index (bool): pandas .to_csv parameter.

    Returns:
        str: Saved full path.
    """

    df.to_csv(save_path, index=index, encoding="utf-8-sig")

    return save_path


def read_intermediate_rates_data() -> pd.DataFrame:
    """Read an intermediate data.

    Returns:
        pd.DataFrame: Read data.
    """

    df_rates = pd.read_csv(INTERMEDIATE_DATA_DIR / INTERMEDIATE_DATA_RATES)
    df_rates = df_rates.set_index("player_name")

    return df_rates


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess raw data.

    Args:
        df (pd.DataFrame): Raw data.

    Returns:
        pd.DataFrame: Preprocessed data.
    """

    # Calculate each rate
    df["first_serve_rate"] = df["first_serve_made"] / df["first_serve_attempted"]
    df["first_serve_point_rate"] = (
        df["first_serve_points_made"] / df["first_serve_points_attempted"]
    )
    df["second_serve_rate"] = (
        1 - df["double_faults"] / df["second_serve_points_attempted"]
    )
    df["second_serve_point_rate"] = (
        df["second_serve_points_made"] / df["second_serve_points_attempted"]
    )
    df["first_serve_return_point_rate"] = (
        df["first_serve_return_points_made"] / df["first_serve_return_points_attempted"]
    )
    df["second_serve_return_point_rate"] = (
        df["second_serve_return_points_made"]
        / df["second_serve_return_points_attempted"]
    )

    df["victory"] = df["player_victory"].replace("t", 1).replace("f", 0)
    
    # Calculate mean of each rate for each player
    get_cols = [
        "player_name",
        "first_serve_rate",
        "first_serve_point_rate",
        "second_serve_rate",
        "second_serve_point_rate",
        "first_serve_return_point_rate",
        "second_serve_return_point_rate",
        "victory",
    ]
    df = df[get_cols]
    df = df.groupby("player_name").agg(["mean", "count"])

    # For analysis, drop player who don't have over 10 wins.
    df = df[(df["victory"]["count"] >= 10) & (df["victory"]["mean"] > 0)]

    # Format the dataframe
    levels = df.columns.levels
    codes = df.columns.codes
    df.columns = [levels[0][i] + "_" + levels[1][j] for i, j in zip(codes[0], codes[1])]
    df = df[[c + "_mean" for c in get_cols[1:]]]
    df.columns = get_cols[1:]

    # Standardize
    df = (df - df.mean()) / df.std()
    df = df.dropna()

    return df


def fit_factor_analysis(x: np.ndarray, n_factor: int) -> np.ndarray:
    """Run factor analysis.

    Args:
        x (np.ndarray): Input data.
        n_factor (int): Factor num.

    Returns:
        np.ndarray: Analyzed data.
    """

    # Run factor analysis
    fa = decomposition.FactorAnalysis(n_components=n_factor).fit(x)

    # Display 因子負荷量 for check
    df_factor_loading = pd.DataFrame(columns=ANALYZE_TARGET_COLS)
    for i in range(n_factor):
        df_factor_loading = df_factor_loading.append(
            pd.Series(
                fa.components_[i], index=ANALYZE_TARGET_COLS, name="factor" + str(i)
            )
        )
    display(df_factor_loading)

    return fa


def plot_factor_analysis_result(
    x: np.ndarray, y: np.ndarray, n_factor: int, fa: np.ndarray
) -> np.ndarray:
    """Plot factor analysis result using each factor score.

    Args:
        x (np.ndarray): Analyzed data.
        y (np.ndarray): Label data.
        n_factor (int): Factor num.
        fa (np.ndarray): Analyzed data.

    Returns:
        np.ndarray: Transformed data.
    """

    transformed = fa.fit_transform(x)

    for i, j in itertools.combinations(np.arange(n_factor), 2):

        plt.figure(figsize=(5, 5))
        plt.axes().add_patch(plt.Circle((0, 0), radius=0.5 * 2, ec="r", fill=False))
        plt.scatter(transformed[:, i], transformed[:, j])
        for k, y_ in enumerate(y):
            plt.annotate(
                y_, xy=(transformed[k, i], transformed[k, j]), size=8, alpha=0.6
            )
        fai = fa.components_[i]
        faj = fa.components_[j]
        for k, c in enumerate(ANALYZE_TARGET_COLS):
            plt.arrow(0, 0, fai[k] * 2, faj[k] * 2, color="r", head_width=0.1, alpha=1)
            plt.text(fai[k] * 2.5, faj[k] * 2.5, c, color="r", fontsize=12)
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
        plt.xlabel("factor" + str(i))
        plt.ylabel("factor" + str(j))
        plt.title("factor" + str(i) + " x " + "factor" + str(j))
        plt.show()

    return transformed


def make_result_df(
    y: np.ndarray,
    transformed: np.ndarray,
    n_factor: int,
    plus_minus_list: List[str],
    victory_rates: np.ndarray,
) -> pd.DataFrame:
    """Make result as a dataframe.

    Args:
        y (np.ndarray): Label data.
        transformed (np.ndarray): Transformed data.
        n_factor (int): Factor num.
        plus_minus_list (List[str]): Direction of strength of each factor.
        victory_rates (np.ndarray): Victory data.

    Returns:
        pd.DataFrame: Made dataframe.
    """

    df_res = pd.DataFrame(
        np.concatenate(
            [np.array(y).reshape(len(y), 1), transformed, victory_rates], axis=1
        ),
        columns=["player_name"]
        + ["factor" + str(i) for i in range(n_factor)]
        + ["victory_rate"],
    )

    # Since it depends on the results of the factor analysis
    # whether the factor is greater in the positive or negative direction,
    # check which direction the factor is greater in from a factor analysis result
    # and format plus or minus.
    for i in range(n_factor):
        if plus_minus_list[i] == "plus":
            df_res["factor" + str(i)] = df_res["factor" + str(i)].astype(float)
        else:
            df_res["factor" + str(i)] = -df_res["factor" + str(i)].astype(float)

    df_res["victory_rate"] = df_res["victory_rate"].astype(float)

    return df_res


def plot_top_bottom_factor(df_res: pd.DataFrame, target_factor: str):
    """Plot top 10 and bottom 10 of given factor.

    Args:
        df_res (pd.DataFrame): Result data.
        target_factor (str): Target factor.
    """

    fig, axs = plt.subplots(ncols=2, figsize=(10, 3))
    plt.subplots_adjust(wspace=0)

    tmp = df_res.sort_values(by=target_factor, ascending=False)
    tmp[:10].plot(kind="bar", x="player_name", y=target_factor, ax=axs[0])
    tmp[-10:].plot(
        kind="bar", x="player_name", y=target_factor, color="tomato", ax=axs[1]
    )

    for i in [0, 1]:
        for tick in axs[i].get_xticklabels():
            tick.set_rotation(50)
        axs[i].set_ylim([-3, 3])
        axs[i].legend_.remove()

    axs[1].set_yticks([])
    plt.suptitle(f"{target_factor} TOP10 & BOTTOM10")
    plt.show()


def show_target_player_factor(df_res: pd.DataFrame, n_factor: int, target_player: str):
    """Print rank of each factor of given target player.

    Args:
        df_res (pd.DataFrame): Result data.
        n_factor (int): Factor num.
        target_player (str): Target player.
    """

    print(target_player)
    for i in range(n_factor):
        rank = np.where(
            df_res.sort_values(by="factor" + str(i), ascending=False)[
                "player_name"
            ].values
            == target_player
        )
        rank = rank[0][0] + 1
        print("factor" + str(i), ":", rank, "th /", len(df_res))
