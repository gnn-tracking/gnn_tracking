import numpy as np
import pandas as pd

from gnn_tracking.analysis.plotutils import Plot
from gnn_tracking.utils.nomenclature import variable_manager as vm


class TracksVsDBSCANPlot(Plot):
    def __init__(self, mean_df: pd.DataFrame, **kwargs):
        """Plot tracking efficiencies vs DBSCAN epsilon

        .. code-block:: python

            tvdp = TracksVsDBSCANPlot(
                mean_df=tcmodule.cluster_scanner.get_results().df_mean,
            )
            secondary_k = 4
            tvdp.plot_var("double_majority_pt0.9", secondary_k=secondary_k)
            tvdp.plot_var("lhc_pt0.9", secondary_k=secondary_k)
            tvdp.plot_var("perfect_pt0.9", secondary_k=secondary_k)
        """
        super().__init__(**kwargs)
        self.df = mean_df.sort_values("eps")
        self.ax.set_xlabel(r"DBSCAN $\varepsilon$")
        self.ax.set_ylabel("Metric")

    def plot_var(self, var: str, *, secondary_k: int = 4, **kwargs):
        """Plot an efficiency.

        Args:
            var: Name of the variable to plot
            secondary_k: Plot a second line with this value of k
            **kwargs: Passed to plot function
        """
        line, *_ = self.ax.errorbar(
            "eps",
            var,
            yerr=f"{var}_std",
            data=self.df[self.df["min_samples"] == 1],
            label=vm[var].latex,
            marker="o",
            **kwargs,
        )
        color = line.get_color()
        if secondary_k:
            self.ax.plot(
                "eps",
                var,
                data=self.df[self.df["min_samples"] == secondary_k],
                marker="",
                color=color,
                ls=":",
                label="_hide",
                **kwargs,
            )


class PerformancePlot(Plot):
    def __init__(
        self,
        xs: np.ndarray,
        df: pd.DataFrame,
        *,
        df_ul: pd.DataFrame | None = None,
        x_label: str = r"$p_T$ [GeV]",
        y_label: str = "Efficiency",
        **kwargs,
    ):
        """Plot efficiencies vs some variable (pt, eta, etc.)

        Args:
            xs (np.ndarray): x values (e.g., pt or eta)
            df (pd.DataFrame): Dataframe with values. Errors should be in columns named with suffix ``_err``.
            df_ul (_type_, optional): Dataframe with values for upper limit. Defaults to None.
            x_label (regexp, optional): x label
            y_label (str, optional): y abel
            **kwargs: Passed to `Plot`
        """
        super().__init__(**kwargs)
        self.df = df
        self.df_ul = df_ul
        self.xs = xs
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self._legend_items = []

    def plot_var(
        self,
        var: str,
        color: str,
        *,
        label: str | None = None,
        plot_ul=True,
    ) -> None:
        """Plot variable

        Args:
            var (str): Name of variable
            color (str): Color
            label (str | None, optional): Label for legend
            plot_ul (bool, optional): Plot upper limit if available
        """
        stairs = self.ax.stairs(var, edges=self.xs, data=self.df, color=color, lw=1.5)
        if self.df_ul is not None and plot_ul:
            self.ax.stairs(
                var, edges=self.xs, data=self.df_ul, color=color, lw=1.5, ls=":"
            )
        mids = (self.xs[:-1] + self.xs[1:]) / 2
        if label is None:
            label = vm[var].latex
        bar = self.ax.errorbar(
            mids,
            var,
            yerr=f"{var}_err",
            ls="none",
            color=color,
            data=self.df,
        )
        self._legend_items.append(((stairs, bar), label))

    def add_blocked(self, a: float, b: float, label="Not trained for") -> None:
        """Used to mark low pt as "not trained for"."""
        span = self.ax.axvspan(
            a, b, alpha=0.3, color="gray", label=label, linestyle="none"
        )
        self._legend_items.append(((span,), label))

    def add_legend(self, **kwargs) -> None:
        all_handles = [item[0] for item in self._legend_items]
        all_labels = [item[1] for item in self._legend_items]
        self.ax.legend(all_handles, all_labels, **kwargs)
