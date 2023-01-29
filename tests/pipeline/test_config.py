from pathlib import Path

from serde.toml import from_toml, to_toml

from vampires_dpp.pipeline.config import FileInput, OutputDirectory


class TestOutputDirectory:
    def test_default_creation(self):
        od = OutputDirectory()
        assert od.output_directory is None

    def test_creation(self):
        od = OutputDirectory(output_directory="/tmp")
        assert od.output_directory == Path("/tmp")

    def test_default_serialize(self):
        od = OutputDirectory()
        s = to_toml(od)
        assert s == ""

    def test_serialize(self):
        od = OutputDirectory(output_directory="/tmp")
        s = to_toml(od)
        assert s == 'output_directory = "/tmp"\n'
