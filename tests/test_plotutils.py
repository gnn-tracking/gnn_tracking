from gnn_tracking.analysis.plotutils import Plot


def test_plot_baseclass(tmp_path):
    p = Plot(watermark="test", model="test")
    p.add_legend()
    p.save(tmp_path / "tmp.pdf")
    p.save()
