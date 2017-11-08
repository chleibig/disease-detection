# Leveraging uncertainty information from deep neural networks for disease detection

Code and models for [Christian Leibig, Vaneeda Allken, Murat Seckin Ayhan, Philipp Berens, Siegfried Wahl (2016)](https://www.biorxiv.org/content/early/2017/10/18/084210)
developed at the ZEISS Vision Science Lab in collaboration with the Berenslab.

## Getting started

If you want to use the Bayesian CNNs for detecting diabetic retinopathy with uncertainty have a look at `disease-detection/example.ipynb`.

To get things running you need a machine with a NVIDIA GPU and install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker#quick-start). The docker image can be built as follows: Clone the repository and `cd` into the folder `disease-detection` and execute:

```bash
docker build -t uncertain-ai-diagnostics -f docker/Dockerfile .
```
Next, start a Docker container:
```bash
nvidia-docker run -it -p 8888:8888 uncertain-ai-diagnostics
```
This will fire up a jupyter notebook server and tell you the URL you have to point your browser to in order to play around with
the example notebook.


## Contact

leibig.christian@gmail.com