Place a Windows FFmpeg shared build in `vendor/ffmpeg/bin`.

Expected contents include DLLs such as:

- `avcodec-*.dll`
- `avformat-*.dll`
- `avutil-*.dll`

Recommended source:

- Gyan "full-shared" Windows build

GPIPA will automatically add `vendor/ffmpeg/bin` to the DLL search path on
Windows during package import. You can also override this location with the
`GPIPA_FFMPEG_BIN` environment variable.
