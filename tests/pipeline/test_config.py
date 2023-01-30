from inspect import cleandoc
from pathlib import Path

import pytest
import tomli
from serde.toml import to_toml

from vampires_dpp.pipeline.config import (
    CalibrateOptions,
    CamFileInput,
    CoronagraphOptions,
    DistortionOptions,
    FileInput,
    OutputDirectory,
)


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
        assert conf.satspot_radius == 15.9
        assert conf.satspot_angle == -4
        toml_conf = CoronagraphOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf

    def test_creation(self):
        conf = CoronagraphOptions(iwa=55, satspot_radius=11.2, satspot_angle=-6.5)
        assert conf.iwa == 55
        assert conf.satspot_radius == 11.2
        assert conf.satspot_angle == -6.5
        toml_conf = CoronagraphOptions(**tomli.loads(to_toml(conf)))
        assert conf == toml_conf
