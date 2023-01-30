from pathlib import Path

import pytest
from serde.toml import from_toml, to_toml

from vampires_dpp.pipeline.config import (
    CalibrateOptions,
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
        s = to_toml(conf)
        assert s == 'output_directory = "/tmp"\n'


class TestFileInput:
    def test_default_creation(self):
        with pytest.raises(TypeError):
            conf = FileInput()

    def test_creation(self):
        conf = FileInput(filenames="/tmp/VMPA*.fits")
        assert conf.filenames == Path("/tmp/VMPA*.fits")

    def test_serialize(self):
        conf = FileInput(filenames="/tmp/VMPA*.fits")
        s = to_toml(conf)
        assert s == 'filenames = "/tmp/VMPA*.fits"\n'


class TestDistortionOptions:
    def test_default_creation(self):
        with pytest.raises(TypeError):
            conf = DistortionOptions()

    def test_creation(self):
        conf = DistortionOptions(transform_filename="/tmp/transforms.csv")
        assert conf.transform_filename == Path("/tmp/transforms.csv")

    def test_serialize(self):
        conf = DistortionOptions(transform_filename="/tmp/transforms.csv")
        s = to_toml(conf)
        assert s == 'transform_filename = "/tmp/transforms.csv"\n'


class TestCalibrateOptions:
    def test_default_creation(self):
        conf = CalibrateOptions()
        def_dict = dict(cam1=None, cam2=None)
        assert conf.darks == def_dict
        assert conf.flats == def_dict
        assert conf.distortion is None
        assert not conf.deinterleave

    def test_creation(self):
        conf = CalibrateOptions(
            output_directory="/tmp/output",
            darks=dict(
                cam1="/tmp/darks/master_dark_cam1.fits", cam2="/tmp/darks/master_dark_cam2.fits"
            ),
            flats=dict(cam1="/tmp/flats/master_flat_cam1.fits", cam2=None),
            distortion=dict(transform_filename="/tmp/transforms.csv"),
        )
        assert conf.darks == dict(
            cam1=Path("/tmp/darks/master_dark_cam1.fits"),
            cam2=Path("/tmp/darks/master_dark_cam2.fits"),
        )
        assert conf.flats == dict(cam1=Path("/tmp/flats/master_flat_cam1.fits"), cam2=None)
        assert conf.distortion == DistortionOptions(transform_filename="/tmp/transforms.csv")
        assert not conf.deinterleave

    def test_serialize(self):
        conf = CalibrateOptions(
            output_directory="/tmp/output",
            darks=dict(
                cam1="/tmp/darks/master_dark_cam1.fits", cam2="/tmp/darks/master_dark_cam2.fits"
            ),
            flats=dict(cam1="/tmp/flats/master_flat_cam1.fits", cam2=None),
            distortion=dict(transform_filename="/tmp/transforms.csv"),
        )
        s = to_toml(conf)
        # assert s == """output_directory = "/tmp/transforms.csv"'
