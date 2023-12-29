
# Frequently Asked Questions (FAQ)

## Pipeline and Configuration version mismatch

> I can't run the pipeline because it says my TOML file has a version mismatch?

In order to try and manage compatibility for the pipeline, your configuration file has a `version` key in it. This key must be compatible (within SemVer) with the installed version of `vampires_dpp`. There are two approaches to fixing this:

1. (Recommended) Call `dpp upgrade` to try to automatically upgrade your configuration
2. Downgrade `vampires_dpp` to match the version in your configuration

## I'm getting warnings about centroid files, help!

The blah blah explain it.

TODO

## Performance

> It's slow. It's so, so slow. Help.

It's hard to process data in the volumes that VAMPIRES produces, but there are some tips for speeding it up.
1. Use an SSD (over USB 3 or thunderbolt)

Faster storage media reduces slowdowns from opening and closing files, which happens *a lot* throughout the pipeline

2. Don't save intermediate files

The time it takes to open a file, write to disk, and close it will add a lot to your overheads, in addition to the huge increase in data volume

3. Use multi-processing

Using more processes should improve some parts of the pipeline, but don't expect multiplicative increases in speed since most operations are limited by the storage IO speed.