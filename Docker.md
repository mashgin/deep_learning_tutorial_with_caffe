
### Cloning the git repo

clone the git repo into your brand new folder and change into that folder

```
git clone https://github.com/mashgin/basic_deep_learning_tutorial.git
cd basic_deep_learning_tutorial
```

### Running the code in Docker

build your docker image (choose a name):

```
sudo docker build -t <image name>
``` 

run `pwd` to get your full working directory path.

run your newly built docker image in a container with acces to those files you just cloned: 

```
sudo docker run  -it -v <copy paste you full working directory path here>:/basic_shapes <image name>
```

you should see something like: `root@CONTAINER:/`. This means you are now inside your container with your environemnt where caffe is installed in.

there's a small bug so run `sudo ln /dev/null /dev/raw1394` after getting that container running, to avoid errors when running the files. 

to make sure caffe is properly working go ahead and run `python`. You should see a prompt as such : 

```
Python 2.7.6 (default, Oct 26 2016, 20:30:19) 
[GCC 4.8.4] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> 
```

go ahead and run `import caffe` if nothing happens except another prompt `>>>`  showing up, you're all set! Run `exit()` to get out of python.

with `ls` you cann see all your directories in your current path. run `cd basic_shapes` to get into the folder with the codes.

Now you can run : `./generate_network.py`, `./train.py`, `./classify.py`, `./classify.py 1`




### Problem getting images to pop up 


if you're having trouble getting the imgaes to pop up with `./classify 1` try:

you have to exit out of your container, run: `xhost -local:root`

then go ahead start the container up again , but passing along the dispaly as well as the xserver directory as such:

`sudo docker run -it --env="DISPLAY"  --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -v <path to your working direcotry>:/basic_shapes <name of image>`





