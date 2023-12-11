
# Frequently Asked Questions (FAQ)

## Pipeline and Configuration version mismatch

> I can't run the pipeline because it says my TOML file has a version mismatch?

In order to try and manage compatibility for the pipeline, your configuration file has a `version` key in it. This key must be compatible (within SemVer) with the installed version of `vampires_dpp`. There are two approaches to fixing this:

1. (Recommended) Call `dpp upgrade` to try to automatically upgrade your configuration
2. Downgrade `vampires_dpp` to match the version in your configuration

## I'm getting warnings about centroid files, help!

The blah blah explain it.

TODO

## I keep getting an error about PrimaryHDUs when re-running the pipeline

If the pipeline crashes or is stopped early there is a chance that data in the process of being written to disk will become corrupted. This appears to happen frequently when saving intermediate calibrated data while multi-processing. To address this error, you will need to replace the corrupted data with a reprocessed version.

To find the file causing the problems, first try looking in the debug log file (`debug.log`) and look for a line describing which file was opened before the exception was thrown. Another way to find a file is to check if the file size matches what is expected (compare it to nearby files' sizes). Lastly, you can force the reprocessing completely through the command line or by removing the intermediate folder, but this will reprocess all files. To reprocess only one file if you have found it, just delete it and re-run the pipeline.


## Performance

> It's slow. It's so, so slow. Help.

It's hard to process data in the volumes that VAMPIRES produces, but there are some tips for speeding it up.
1. Use an SSD (over USB 3 or thunderbolt)

Faster storage media reduces slowdowns from opening and closing files, which happens *a lot* throughout the pipeline

2. Don't save intermediate files

The time it takes to open a file, write to disk, and close it will add a lot to your overheads, in addition to the huge increase in data volume

3. Use multi-processing

Using more processes should improve some parts of the pipeline, but don't expect multiplicative increases in speed since most operations are limited by the storage IO speed.