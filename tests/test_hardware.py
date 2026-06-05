from hearth import hardware


def test_probe_returns_shape():
    hw = hardware.probe()
    assert hw.cpu_cores is None or hw.cpu_cores > 0
    assert isinstance(hw.gpus, list)


def test_as_dict_has_inference_target():
    d = hardware.as_dict()
    assert "inference_target" in d
    assert "gpus" in d and "ram_total_mib" in d
