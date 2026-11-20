from easplatting.rescore import print_table


def test_print_table_groups_by_method(capsys):
    row = dict(method="ga", kept=0.5, fit_psnr=40.1, floors_ok=True)
    row.update({"fit-full_mean": 40.0, "fit-full_worst": 36.0, "test_mean": 40.2, "test_worst": 35.5})
    bad = dict(row, method="cmaes", floors_ok=False)
    print_table([row, row, bad], dict(scene="x", tau=40.0, worst_tau=35.0))
    out = capsys.readouterr().out
    assert "2/2" in out and "0/1" in out
