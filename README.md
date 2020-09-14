# MRI Reconstruction

A project developed as a part of Praktikum Machine Learning in Medical Imaging by
- Devendra Vyas
- Fabian Groeger
- Maik Dannecker
- Viktor Studenyak

Mentored by : Shahrooz Roohi

## How to Run
### Locally
TO run locally on your system simply go to the root directory and use the `run` scrip as:
```bash
my_user@my_device: ./run.sh --model=WNET
```
Or you can directly run the `main_train.py` using:
```bash
my_user@my_device: python main_train.py --model=WNET
```
The argument `--model` can take either of the 2 values {'WNET', 'RL'}

### On Google Cloud
1. Install Google Cloud SDK
2. Go to console.cloud.google.com
3.  Start setting up the Google Cloud SDK locally. Open your terminal and type in `gcloud init` 
```bash
skat00sh@XPS:~/projects/mri-reconstruction$ gcloud init
Welcome! This command will take you through the configuration of gcloud.
```
You'll get your first prompt. Choose the option to create a new configuration
```
Pick configuration to use:
 [1] Re-initialize this configuration [default] with new settings 
 [2] Create a new configuration
Please enter your numeric choice:  2
```
It then asks you to set up a confiuration name. This is just for you, so that in case you have multiple configurations setup on your system you can switch easily. For example `mri-challenge` 
```
Enter configuration name. Names start with a lower case letter and 
contain only lower case letters a-z, digits 0-9, and hyphens '-':  mri-challenge
Your current configuration has been set to: [mri-challenge]
```
You then would be asked to Login with a account. If you already have an account configured
```
You can skip diagnostics next time by using the following flag:
  gcloud init --skip-diagnostics

Network diagnostic detects and fixes local network connection issues.
Checking network connection...done.                                                                                                                                                          
Reachability Check passed.
Network diagnostic passed (1/1 checks passed).

Choose the account you would like to use to perform operations for 
this configuration:
 [1] vyas.dev.ms@gmail.com
 [2] Log in with a new account
Please enter your numeric choice:  2

Your browser has been opened to visit:

    https://accounts.google.com/o/oauth2/auth?code_challenge=5bqNhwa0-HS0jAbhP7yED7x6H6s8v_IvfgjR4-kbgGQ&prompt=select_account&code_challenge_method=S256&access_type=offline&redirect_uri=http%3A%2F%2Flocalhost%3A8085%2F&response_type=code&client_id=32555940559.apps.googleusercontent.com&scope=openid+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fuserinfo.email+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcloud-platform+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fappengine.admin+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcompute+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Faccounts.reauth


You are logged in as: [tummcmrirecon@gmail.com].

Pick cloud project to use: 
 [1] mri-recon
 [2] Create a new project
Please enter numeric choice or text value (must exactly match list 
item):  1

Your current project has been set to: [mri-recon].

Do you want to configure a default Compute Region and Zone? (Y/n)?  n

Created a default .boto configuration file at [/home/skat00sh/.boto]. See this file and
[https://cloud.google.com/storage/docs/gsutil/commands/config] for more
information about configuring Google Cloud Storage.
Your Google Cloud SDK is configured and ready to use!

* Commands that require authentication will use tummcmrirecon@gmail.com by default
* Commands will reference project `mri-recon` by default
Run `gcloud help config` to learn how to change individual settings

This gcloud configuration is called [mri-challenge]. You can create additional configurations if you work with multiple accounts and/or projects.
Run `gcloud topic configurations` to learn more.

Some things to try next:

* Run `gcloud --help` to see the Cloud Platform services you can interact with. And run `gcloud help COMMAND` to get help on any gcloud command.
* Run `gcloud topic --help` to learn about advanced features of the SDK like arg files and output formatting

   ```
1. Click on the SSH dropdown icon as shown in figure and select

![step1](assets/step1.png)

## Configurations
### For W-net
Description for the properties used in `config_wnet.py` file:


```python
data_root:  `<Path to root folder that contains Train and Val data>`,  
challenge: Depending on type of data calue could be `multicoil` OR `singlecoil` 
dim: Dimension of input kspace | Default = (218, 170)
domain: I/O domain. Could be one of the following values {`ki`, `ii`, `ik`, `kk`}
sample_rate:Sample rate of input data
sampling_dist: Sampling distribution undersampling. Could be one of the following values {`poisson`, `gaussian`, `uniform`}
norm: Boolean value to compute the Root Sum of Squares (RSS)
slice_cut: Crop (first and last) slices during read in | Deafult = (50, 50)
batch_size: Batch size for each iteration
num_workers: Denotes the number of processes that generate batches in parallel | Deafult = 3
```

## Contribution Guidelines
Please follow the guidelines mentioned in [contributing.md](contributing.md) for smooth collaboration :) 
