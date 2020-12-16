# MRI Reconstruction

A project developed as a part of Praktikum Machine Learning in Medical Imaging at the Technical University of Munich by
- [Devendra Vyas](https://github.com/skat00sh)
- Fabian Groeger
- Maik Dannecker
- Viktor Studenyak

Mentored by : Shahrooz Roohi

This work was used in the [Multi-channel MR Image Reconstruction Challenge (MC-MRRec)](https://sites.google.com/view/calgary-campinas-dataset/mr-reconstruction-challenge) as part of the 2020 Medical Imaging with Deep Learning (MIDL) conference.

## How to Run
To run on your system simply go to the root directory and use the `run` scrip as:
```bash
my_user@my_device: ./run.sh --model=WNET
```
Or you can directly run the `main_train.py` using:
```bash
my_user@my_device: python main_train.py --model=WNET
```
The argument `--model` can take either of the 2 values {'WNET', 'RL'}
