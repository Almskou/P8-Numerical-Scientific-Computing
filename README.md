# P8-Numerical-Scientific-Computing
Files for the mini project in the course Numerical Scientific Computing at Aalborg University

Pre calculated data is found in the data folder for following resolutions: 100, 500, 1000, 2000. 
Mandlebrot plots can also be found in the data folder for following resolutions: 100, 500, 1000, 2000, 5000.

To calculate the mandelbrot set run the `Mandelbrot_create_data.py`
The resolution and number of cores/workers can also be specified in the file.

After this the data will be saved to multiple files in created folders.
In order to plot the mandelbrot sets calculated run the `Mandelbrot_plot.py` file from the same directory the `Mandelbrot_create_data.py` was run from.
If resolution or number of cores/workers is changed in `Mandelbrot_create_data.py` corrosponding change should also be made in `Mandelbrot_plot.py`. This includes changing the resolution, new subfolder names in `data/` and the title of the data files. 

The same changes should be done if it is wanted to plot the time plots in `Mandelbrot_plot_time.py`.
