from gnn_tracking.utils.nomenclature import variable_manager as vm


def test_variable_manager():
    assert vm["eta"].latex == r"$\eta$"
    assert vm["does_not_exist"].latex == "does_not_exist"
