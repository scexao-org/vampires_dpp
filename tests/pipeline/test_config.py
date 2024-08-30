from pathlib import Path

import pytest
import tomli
from vampires_dpp.pipeline.config import (
    CalibrateOptions,
    CamFileInput,
    CollapseOptions,
    CoronagraphOptions,
    DistortionOptions,
    FrameSelectOptions,
    IPOptions,
    PipelineOptions,
    PolarimetryOptions,
    RegisterOptions,
    SatspotOptions,
    to_toml,
)


class TestCalibrateOptions:
    def test_default_creation(self):
        conf = CalibrateOptions()
        assert conf.master_darks == CamFileInput()
        assert conf.master_flats == CamFileInput()
        assert conf.distortion is None
        assert not conf.deinterleave

    def test_creation(self, tmp_path):
        conf = CalibrateOptions(
            output_directory=tmp_path / "output",
            master_darks=dict(
                cam1=tmp_path / "darks" / "master_dark_cam1.fits",
                cam2=tmp_path / "darks" / "master_dark_cam2.fits",
            ),
            master_flats=dict(cam1=tmp_path / "flats" / "master_flat_cam1.fits", cam2=None),
            distortion=dict(transform_filename=tmp_path / "transforms.csv"),
        )
        assert conf.master_darks == CamFileInput(
            cam1=tmp_path / "darks" / "master_dark_cam1.fits",
            cam2=tmp_path / "darks" / "master_dark_cam2.fits",
        )
        assert conf.master_flats == CamFileInput(cam1=tmp_path / "flats" / "master_flat_cam1.fits")
        assert conf.distortion == DistortionOptions(transform_filename=tmp_path / "transforms.csv")
        assert not conf.deinterleave

    def test_serialize(self, tmp_path):
        conf = CalibrateOptions(
            output_directory=tmp_path / "output",
            master_darks=dict(
                cam1=tmp_path / "darks" / "master_dark_cam1.fits",
                cam2=tmp_path / "darks" / "master_dark_cam2.fits",
            ),
            master_flats=dict(cam1=tmp_path / "flats" / "master_flat_cam1.fits", cam2=None),
            distortion=dict(transform_filename=tmp_path / "transforms.csv"),
        )
        toml_conf = CalibrateOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf


class TestCoronagraphOptions:
    def test_error_creation(self):
        with pytest.raises(TypeError):
            CoronagraphOptions()

    def test_default_creation(self):
        conf = CoronagraphOptions(iwa=55)
        assert conf.iwa == 55
        toml_conf = CoronagraphOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf


class TestSatspotOptions:
    def test_default_creation(self):
        conf = SatspotOptions()
        assert conf.radius == 15.9
        assert conf.angle == -5.4
        assert conf.amp == 50
        toml_conf = SatspotOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf

    def test_creation(self):
        conf = SatspotOptions(radius=11.2, angle=-6.5, amp=25)
        assert conf.radius == 11.2
        assert conf.angle == -6.5
        assert conf.amp == 25
        toml_conf = SatspotOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf


class TestFrameSelectOptions:
    def test_error_creation(self):
        with pytest.raises(TypeError):
            FrameSelectOptions()

    def test_default_creation(self):
        conf = FrameSelectOptions(cutoff=0.2)
        assert conf.cutoff == 0.2
        assert conf.metric == "normvar"
        assert conf.window_size == 31
        assert not conf.force
        assert conf.output_directory is None
        toml_conf = FrameSelectOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf

    @pytest.mark.parametrize("metric", ["peak", "l2norm", "normvar"])
    def test_creation(self, metric, tmp_path):
        conf = FrameSelectOptions(
            cutoff=0.2, metric=metric, window_size=20, force=True, output_directory=tmp_path
        )
        assert conf.cutoff == 0.2
        assert conf.metric == metric
        assert conf.window_size == 20
        assert conf.force
        assert conf.output_directory == tmp_path
        toml_conf = FrameSelectOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf


class TestRegisterOptions:
    def test_default_creation(self):
        conf = RegisterOptions()
        assert conf.method == "com"
        assert conf.window_size == 31
        assert not conf.force
        assert conf.output_directory is None
        toml_conf = RegisterOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf

    @pytest.mark.parametrize("method", ["peak", "com", "dft", "airydisk", "moffat", "gaussian"])
    def test_creation(self, method, tmp_path):
        conf = RegisterOptions(method=method, window_size=20, force=True, output_directory=tmp_path)
        assert conf.method == method
        assert conf.window_size == 20
        assert conf.force
        assert conf.output_directory == tmp_path
        toml_conf = RegisterOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf


class TestCollapseOptions:
    def test_default_creation(self):
        conf = CollapseOptions()
        assert conf.method == "median"
        assert not conf.force
        assert conf.output_directory is None
        toml_conf = CollapseOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf

    @pytest.mark.parametrize("method", ["median", "mean", "varmean", "biweight"])
    def test_creation(self, method, tmp_path):
        conf = CollapseOptions(method=method, force=True, output_directory=tmp_path)
        assert conf.method == method
        assert conf.force
        assert conf.output_directory == Path(tmp_path)
        toml_conf = CollapseOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf


class TestPolarimetryOptions:
    def test_default_creation(self):
        conf = PolarimetryOptions()
        assert not conf.force
        assert conf.output_directory is None
        toml_conf = PolarimetryOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf

    def test_creation(self, tmp_path):
        conf = PolarimetryOptions(force=True, output_directory=tmp_path)
        assert conf.force
        assert conf.output_directory == Path(tmp_path)
        toml_conf = PolarimetryOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf


class TestIPOptions:
    def test_default_creation(self):
        conf = IPOptions()
        assert conf.method == "photometry"
        assert conf.aper_rad == 6
        toml_conf = IPOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf

    @pytest.mark.parametrize("method", ["photometry", "satspots", "mueller"])
    def test_creation(self, method):
        conf = IPOptions(method=method, aper_rad=8)
        assert conf.method == method
        assert conf.aper_rad == 8
        toml_conf = IPOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf


class TestPipelineOptions:
    def test_error_creation(self):
        with pytest.raises(TypeError):
            PipelineOptions()

    def test_default_creation(self, tmp_path):
        conf = PipelineOptions(filenames=tmp_path / "VMPA*.fits", name="test")
        assert conf.name == "test"
        assert conf.frame_centers is None
        assert conf.target is None
        assert conf.coronagraph is None
        assert conf.calibrate is None
        assert conf.frame_select is None
        assert conf.register is None
        assert conf.collapse is None
        assert conf.polarimetry is None
        toml_conf = PipelineOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf

    def test_creation(self, tmp_path):
        conf = PipelineOptions(
            filenames=tmp_path / "VMPA*.fits",
            name="test",
            target="AB Aur",
            coronagraph=dict(iwa=55),
            satspots=dict(radius=11.2),
            calibrate=dict(
                output_directory="calibrated",
                master_darks=dict(cam1=tmp_path / "darks" / "master_dark_cam1.fits"),
            ),
            frame_select=dict(cutoff=0.3, output_directory="selected"),
            register=dict(method="com", output_directory="aligned"),
            collapse=dict(output_directory="collapsed"),
            polarimetry=dict(output_directory="pdi"),
        )
        assert conf.name == "test"
        assert conf.frame_centers is None
        assert conf.target == "AB Aur"
        assert conf.coronagraph == CoronagraphOptions(55)
        assert conf.satspots == SatspotOptions(radius=11.2)
        assert conf.calibrate == CalibrateOptions(
            output_directory="calibrated",
            master_darks=dict(cam1=tmp_path / "darks" / "master_dark_cam1.fits"),
        )
        assert conf.frame_select == FrameSelectOptions(cutoff=0.3, output_directory="selected")
        assert conf.register == RegisterOptions(method="com", output_directory="aligned")
        assert conf.collapse == CollapseOptions(output_directory="collapsed")
        assert conf.polarimetry == PolarimetryOptions(output_directory="pdi")
        toml_conf = PipelineOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf
