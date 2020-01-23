# LibRaw fork

I'm messing with Libraw to try and add diagnostic info.

Build & run:

```
make -f Makefile.devel bin/dcraw_emu && bin/dcraw_emu -W -w -o 0 -v -v -v ~/Downloads/raw/ROFL0296.raf
```

## dcraw_emu options

Just run `dcraw_emu` without flags, but for reference:

-W -- no auto-brighten
-w -- camera white balance
-o 0 -- use camera colorspace
-v -v -v -- verbose verbose verbose
-q 2 -- use 1-pass Frank's xtrans algorithm
-q 1 -- use linear maybe?
