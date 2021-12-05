# FashionMNIST_G25
Image classification based on the "Fashion MNIST" dataset for the Machine Learning course

# Structure of code
- All the configurations and main running stuff : runner.py
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
