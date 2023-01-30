from pathlib import Path

import pytest
from serde.toml import from_toml, to_toml

from vampires_dpp.pipeline.config import DistortionOptions, FileInput, OutputDirectory


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
