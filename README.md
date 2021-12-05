# FashionMNIST_G25
Image classification based on the "Fashion MNIST" dataset for the Machine Learning course

# DATA (IMPORTANT)
- Download the data from https://www.kaggle.com/zalando-research/fashionmnist
- Put ALL of the files in the data folder. All the .csv and ubyte stuff. Or nothing will happen!

# Structure of code
- All the configurations and main running stuff : runner.py
  - I patched in a way to just take a subset of the data for faster computation : Just go to runner.py, in the load_data function there just add whatever number of images you want for subset. Eg 1000.
- All the functions : backbone.py
- The .ipynb jupyter notebooks are auto generated but you can use them to test new features etc
  - They WILL be overwritten. So make sure to either rename it to something else or save them elsewhere to be safe.

# How to get it working
- First install all the requirements using ```pip install -r requirements.txt```
- python3 runner.py should run everything youve set up
- If you are using WSL/Linux : please first do 
  - chmod +x pusher.sh (Just once)
  - Everytime you want to save to github : ```./pusher.sh "commit_message"```
    - This runs a bunch of code formatting stuff in the background
