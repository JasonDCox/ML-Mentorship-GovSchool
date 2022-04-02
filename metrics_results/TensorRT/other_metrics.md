## FPS (includes converting input array, running model, and turning output into usable dictionary list)
Average: 3.695955 (includes first image out of the testing, which typically takes significantly longer)

## Memory usage
program usage (tracemalloc): 1061 MB

Jetson usage (tegrastats): ~1850 MB RAM ~3900 MB Swap

pre-idle usage (tegrastats): 691 MB RAM 158 MB Swap
post-idle usage (tegrastats): 732 MB RAM 346 MB Swap

## CPU usage
Loading usage (tegrastats): all over the place, but rather low
Jetson usage (tegrastats): ~50% all core

## GPU usage
Loading usage (tegrastats): practically none
Jetson usage (tegrastats): 100%

## TF-TRT load in and preperation time (includes loading tensorflow and loading the saved (already optimized) model)
23 minutes and 15 seconds
