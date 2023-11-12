import pytest
from vampires_dpp.constants import CMOSVAMPIRES, EMCCDVAMPIRES


@pytest.mark.parametrize(
    "inst",
    (
        EMCCDVAMPIRES(cam_num=1, emgain=0),
        EMCCDVAMPIRES(cam_num=2, emgain=0),
        EMCCDVAMPIRES(cam_num=1, emgain=300),
        EMCCDVAMPIRES(cam_num=2, emgain=300),
        CMOSVAMPIRES(cam_num=1, readmode="fast"),
        CMOSVAMPIRES(cam_num=2, readmode="fast"),
        CMOSVAMPIRES(cam_num=1, readmode="slow"),
        CMOSVAMPIRES(cam_num=2, readmode="slow"),
    ),
)
def test_instrument_info(inst):
    assert inst.readnoise
    assert inst.gain
    assert inst.pixel_scale
    assert inst.pa_offset
    assert inst.pupil_offset
    assert inst.fullwell
