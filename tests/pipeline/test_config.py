from pathlib import Path

import pytest
import tomli
from serde.toml import to_toml

from vampires_dpp.pipeline.config import *


class TestOutputDirectory:
    def test_default_creation(self):
        conf = OutputDirectory()
        assert conf.output_directory is None

    def test_creation(self):
        conf = OutputDirectory(output_directory="/tmp")
        assert conf.output_directory == Path("/tmp")

    def test_default_serialize(self):
        conf = OutputDirectory()
        s = to_toml(conf)
        assert s == ""

    def test_serialize(self):
        conf = OutputDirectory(output_directory="/tmp")
        toml_conf = OutputDirectory(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf


class TestFileInput:
    def test_default_creation(self):
        with pytest.raises(TypeError):
            conf = FileInput()

    def test_creation(self):
        conf = FileInput(filenames="/tmp/VMPA*.fits")
        assert conf.filenames == Path("/tmp/VMPA*.fits")

    def test_serialize(self):
        conf = FileInput(filenames="/tmp/VMPA*.fits")
        toml_conf = FileInput(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf


class TestDistortionOptions:
    def test_default_creation(self):
        with pytest.raises(TypeError):
            conf = DistortionOptions()

    def test_creation(self):
        conf = DistortionOptions(transform_filename="/tmp/transforms.csv")
        assert conf.transform_filename == Path("/tmp/transforms.csv")

    def test_serialize(self):
        conf = DistortionOptions(transform_filename="/tmp/transforms.csv")
        toml_conf = DistortionOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf


class TestCalibrateOptions:
    def test_default_creation(self):
        conf = CalibrateOptions()
        assert conf.master_darks == CamFileInput()
        assert conf.master_flats == CamFileInput()
        assert conf.distortion is None
        assert not conf.deinterleave

    def test_creation(self):
        conf = CalibrateOptions(
            output_directory="/tmp/output",
            master_darks=dict(
                cam1="/tmp/darks/master_dark_cam1.fits", cam2="/tmp/darks/master_dark_cam2.fits"
            ),
            master_flats=dict(cam1="/tmp/flats/master_flat_cam1.fits", cam2=None),
            distortion=dict(transform_filename="/tmp/transforms.csv"),
        )
        assert conf.master_darks == CamFileInput(
            cam1="/tmp/darks/master_dark_cam1.fits",
            cam2="/tmp/darks/master_dark_cam2.fits",
        )
        assert conf.master_flats == CamFileInput(cam1="/tmp/flats/master_flat_cam1.fits")
        assert conf.distortion == DistortionOptions(transform_filename="/tmp/transforms.csv")
        assert not conf.deinterleave

    def test_serialize(self):
        conf = CalibrateOptions(
            output_directory="/tmp/output",
            master_darks=dict(
                cam1="/tmp/darks/master_dark_cam1.fits", cam2="/tmp/darks/master_dark_cam2.fits"
            ),
            master_flats=dict(cam1="/tmp/flats/master_flat_cam1.fits", cam2=None),
            distortion=dict(transform_filename="/tmp/transforms.csv"),
        )
        toml_conf = CalibrateOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf


class TestCoronagraphOptions:
    def test_error_creation(self):
        with pytest.raises(TypeError):
            conf = CoronagraphOptions()

    def test_default_creation(self):
        conf = CoronagraphOptions(iwa=55)
        assert conf.iwa == 55
        toml_conf = CoronagraphOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf


class TestSatspotOptions:
    def test_default_creation(self):
        conf = SatspotOptions()
        assert conf.radius == 15.9
        assert conf.angle == -4
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
            conf = FrameSelectOptions()

    def test_default_creation(self):
        conf = FrameSelectOptions(cutoff=0.2)
        assert conf.cutoff == 0.2
        assert conf.metric == "normvar"
        assert conf.window_size == 30
        assert not conf.force
        assert conf.output_directory is None
        toml_conf = FrameSelectOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf

    def test_creation(self):
        conf = FrameSelectOptions(
            cutoff=0.2, metric="l2norm", window_size=20, force=True, output_directory="/tmp"
        )
        assert conf.cutoff == 0.2
        assert conf.metric == "l2norm"
        assert conf.window_size == 20
        assert conf.force
        assert conf.output_directory == Path("/tmp")
        toml_conf = FrameSelectOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf


class TestCoregisterOptions:
    def test_default_creation(self):
        conf = CoregisterOptions()
        assert conf.method == "com"
        assert conf.window_size == 30
        assert not conf.force
        assert conf.output_directory is None
        toml_conf = CoregisterOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf

    def test_creation(self):
        conf = CoregisterOptions(method="peak", window_size=20, force=True, output_directory="/tmp")
        assert conf.method == "peak"
        assert conf.window_size == 20
        assert conf.force
        assert conf.output_directory == Path("/tmp")
        toml_conf = CoregisterOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf


class TestCollapseOptions:
    def test_default_creation(self):
        conf = CollapseOptions()
        assert conf.method == "median"
        assert not conf.force
        assert conf.output_directory is None
        toml_conf = CollapseOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf

    def test_creation(self):
        conf = CollapseOptions(method="varmean", force=True, output_directory="/tmp")
        assert conf.method == "varmean"
        assert conf.force
        assert conf.output_directory == Path("/tmp")
        toml_conf = CollapseOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf


class TestPolarimetryOptions:
    def test_default_creation(self):
        conf = PolarimetryOptions()
        assert not conf.force
        assert conf.output_directory is None
        toml_conf = PolarimetryOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf

    def test_creation(self):
        conf = PolarimetryOptions(force=True, output_directory="/tmp")
        assert conf.force
        assert conf.output_directory == Path("/tmp")
        toml_conf = PolarimetryOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf


class TestPipelineOptions:
    def test_error_creation(self):
        with pytest.raises(TypeError):
            conf = PipelineOptions()

    def test_default_creation(self):
        conf = PipelineOptions(filenames="/tmp/VMPA*.fits", name="test")
        assert conf.name == "test"
        assert conf.frame_centers is None
        assert conf.target is None
        assert conf.coronagraph is None
        assert conf.calibrate is None
        assert conf.frame_select is None
        assert conf.coregister is None
        assert conf.collapse is None
        assert conf.polarimetry is None
        toml_conf = PipelineOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf

    def test_creation(self):
        conf = PipelineOptions(
            filenames="/tmp/VMPA*.fits",
            name="test",
            target="AB Aur",
            coronagraph=dict(iwa=55),
            satspot=dict(radius=11.2),
            calibrate=dict(
                output_directory="calibrated", master_darks=dict(cam1="/tmp/master_dark_cam1.fits")
            ),
            frame_select=dict(cutoff=0.3, output_directory="selected"),
            coregister=dict(method="com", output_directory="aligned"),
            collapse=dict(output_directory="collapsed"),
            polarimetry=dict(output_directory="pdi"),
        )
        assert conf.name == "test"
        assert conf.frame_centers is None
        assert conf.target == "AB Aur"
        assert conf.coronagraph == CoronagraphOptions(55)
        assert conf.satspot == SatspotOptions(radius=11.2)
        assert conf.calibrate == CalibrateOptions(
            output_directory="calibrated", master_darks=dict(cam1="/tmp/master_dark_cam1.fits")
        )
        assert conf.frame_select == FrameSelectOptions(cutoff=0.3, output_directory="selected")
        assert conf.coregister == CoregisterOptions(method="com", output_directory="aligned")
        assert conf.collapse == CollapseOptions(output_directory="collapsed")
        assert conf.polarimetry == PolarimetryOptions(output_directory="pdi")
        toml_conf = PipelineOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf
