# A basic deep learning tutorial with caffe

A basic tutorial on how to understand your first neural net to classify black and white images of circles and squares using caffe. Images like these:

![circle](/examples/circ_1.png) 		![square](/examples/squ_1.png)

If concepts like neural nets, layers, nodes, backpropagation ... are completely new to you I recommend having a look at this [tutorial](http://adilmoujahid.com/posts/2016/06/introduction-deep-learning-python-caffe/). Chapters 1, 2 and 3 helped me out a lot. For those of you who have more time or are more serious about getting into deep learning this [book](http://neuralnetworksanddeeplearning.com/) is very well written. 

### Getting Caffe running in Docker

I would strongly advise you to get Caffe working with Docker. Go ahead and install [Docker](https://www.docker.com/products/docker#). Create an image with the Docker file in the repo, run a container with the image and the python files from the repo mounted in the container. If you're having trouble with this see the [Docker.md](https://github.com/mashgin/deep_learning_tutorial_with_caffe/blob/master/Docker.md) for more detailed instructions.

Now it's time to build your first neural net, train it and use it to distinguish images of circles and squares!

Without further a do, let’s just get started!

### Getting the big picture (Running the programs)

You can run the 3 files as follows :

```
+ First, generate network definitions by running:
    ./generate_network.py

+ Second, train a network:
    ./train.py

+ Check accuracy of trained network:
    ./classify.py

+ View live network classification:
    ./classify.py 1
```

Go ahead and run the first command `./generate_network.py` .

In your terminal you will see the Network topology. This is showing you the size of your network layers: 

```
================ Network topology ================
data (1, 1, 100, 100)
conv1 (1, 16, 96, 96)
pool1 (1, 16, 32, 32)
norm1 (1, 16, 32, 32)
fc1 (1, 128)
score_all (1, 2)
score (1, 2, 1)
loss (1, 2, 1)
==================================================
```
For example the data layer is 4 dimensional. you can think of it as matrix with a size of 1x1x100x100.

Also you will notice a new folder *models* pop up in your working direcotry. If you have a look into that folder `cd models` , you can see that 4 files are created:

* `train.proto` → stores the training net architectur. 

* `test.proto` → stores the test net architecture.

* `deploy.proto` → exactly the same thing as `test.proto`, but this architecture is needed during the classification phase, not the training phase.

* `solver.proto` → storing information we will need to train our network.

*  (after executing `./train.py`, `weights.proto` will appear as well → stores the weights ('knowledge') learned.) 

These files are [protobuf files](https://developers.google.com/protocol-buffers/docs/overview). Similar to JSON files.

go back to your original folder with `cd ..`.

At this point all we have is an empty net architecture. You can imagine it as a new born baby. So let’s teach it something!

After running `./train.py` you will see a bunch of iterations fly by and some testing going on inbetween. What’s happening is you are feeding your net batches of training data, which it uses to learn. After each 100 iterations we are testing how accurate our network already is. Indicated as: `Accuracy: 46.00%`.

Then, we execute `./classify.py.` This will show you to what level of accuracy it was able to learn during training. (hopefully : `Accuracy: 100.00%`) 

If you execute `./classify.py 1` you can see first hand the images we want to classify and the networks output. (Press any key to continue to the next image). If you're having trouble getting the images to pop up see the [Docker.md](https://github.com/mashgin/deep_learning_tutorial_with_caffe/blob/master/Docker.md) for a cure ;). 

Now that we roughly know what’s happening let’s actually build the Neural Net and look at the code. 

### Constructing the Network (generate_network.py)

For best results skim over the `generate_network.py` file before continuing. 

At the bottom of the `generate_network.py` file we find the main function: 

```python

if __name__ == '__main__':

    if not os.path.isdir('models'):
        os.mkdir('models')

    WriteModel() #defining net structure
    WriteSolver() #defining our learning method

    print "================ Network topology ================"
    ShowNet() #printing net dimensions to terminal
    print "=================================================="
```
We are simply creating the *models* folder, defining our net architectures (`WriteModel()`), defining our learning structure (`WriteSolver()`) and then printing the size of those nets to Terminal (`ShowNet()`).  

Let’s have a look at `WriteModel()` and `WriteSolver()`:

```python
def WriteModel():
    '''
        Generate train, test and deploy nets with different params. Save them to file.
        deploy == test
    '''
    with open('models/train.proto', 'w') as f:
        f.write(GetNetwork('train', 64))
    with open('models/test.proto', 'w') as f:
        f.write(GetNetwork('test', 1))
    with open('models/deploy.proto', 'w') as f:
        f.write(GetNetwork('deploy', 1))
```

```python
def WriteSolver():
    '''
        Generate and save the solver params
    '''
    with open('models/solver.proto', 'w') as f:
        f.write(str(GetSolver()))
```
Alright nothing exciting here. Just writing to files. But! These are the protobuf files I was talking about. So how exactly are we creating these net architectures that live in these protobuf files? That my friends is interesting! Let's have a look at how it's done with `GetNetwork()` and `GetSolver()`.

```python
def GetNetwork(mode, batch_size):
    '''
        Network definition
    '''
    n = caffe.NetSpec()

    n.data = L.Input(input_param={'shape': {'dim': [batch_size, 1, n_rows, n_cols]}})
    if mode == 'train':
        n.label = L.Input(input_param={'shape': {'dim': [batch_size, n_classifiers]}})

    n.conv1 = L.Convolution(n.data, kernel_size=5, stride=1, num_output=16, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.conv1, in_place=True)
    n.pool1 = L.Pooling(n.relu1, kernel_size=3, stride=3, pool=P.Pooling.MAX)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)

    n.fc1 = L.InnerProduct(n.norm1, num_output=64, weight_filler=dict(type='xavier'))
    n.fcrelu1 = L.ReLU(n.fc1, in_place=True)
    n.drop1 = L.Dropout(n.fcrelu1, dropout_ratio=0.25, in_place=True)

    n.score_all = L.InnerProduct(n.drop1, num_output=n_classes * n_classifiers, weight_filler=dict(type='xavier'))
    n.score = L.Reshape(n.score_all, reshape_param={'shape': {'dim': [batch_size, n_classes, n_classifiers]}})

    if mode == 'train':
        n.loss = L.SoftmaxWithLoss(n.score, n.label)
    else:
        n.loss = L.Softmax(n.score)

    return str(n.to_proto())
```


Eventhough this is a lot to digest, this IS the heart of the program. But no fear we’ll go through this step by step! In the end, if you understand what’s going on here you’ve come a long way!

So now let’s break this down. 

```python
def GetNetwork(mode, batch_size):
```

The information this function is getting is `mode` and `batch_size`. `mode` indicating which phase (training, testing, classifying) am i defining this net architecture for and `batch_size` being the parameter to define how many pieces of information (images) the network will be expecting. For example, the training net will receive 64 images at a time, where as the testing net will only receive 1 image at a time.

The next thing is kind of a wrapper for all the juiciness: 

```python
n = caffe.NetSpec()

# much scary looking stuff :O

return str(n.to_proto())
```

In order to save the definition of our net to a protobuf file we need a string. So caffe gives us the tools to do this. `caffe.NetSpec()` you can think of initializing a dictionary in python. With `n` now being our dictionary. Filling this dictionary with information is whats happening in the `# scary looking stuff :O` and `str(n.to_proto)` is caffe's tool for formating this dictionary to a string for us. 

Ok time to get down to business. We now want to define our network architecture with layers. Here is where we get into the `# scary looking stuff :O`, which actually is not that scary. If you notice, all of these lines of code are actually pretty similar. That's becasue each of them is doing the same thing: adding a layer to our network. 

Let's take a step back here and look at an image for a second. Notice how there are input, hidden and output layers?  

![Basic Neural Net diagram](http://neuralnetworksanddeeplearning.com/images/tikz41.png)

Here's another example of how such a network could look like from its structure. This is just to give you an idea of how these networks can differ. The reason I'm showing you this, is because the structure of these networks is what all the hype is about at the moment.

![Second Example Basic Neural Net diagram](http://neuralnetworksanddeeplearning.com/images/tikz13.png)


So let's just go ahead and break our layers down into input layer, hidden layers and output layers.  


```python
#input layer

    n.data = L.Input(input_param={'shape': {'dim': [batch_size, 1, n_rows, n_cols]}})
    if mode == 'train':
        n.label = L.Input(input_param={'shape': {'dim': [batch_size, n_classifiers]}})
        
```

```python
#hidden layers

    n.conv1 = L.Convolution(n.data, kernel_size=5, stride=1, num_output=16, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.conv1, in_place=True)
    n.pool1 = L.Pooling(n.relu1, kernel_size=3, stride=3, pool=P.Pooling.MAX)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)

    n.fc1 = L.InnerProduct(n.norm1, num_output=64, weight_filler=dict(type='xavier'))
    n.fcrelu1 = L.ReLU(n.fc1, in_place=True)
    n.drop1 = L.Dropout(n.fcrelu1, dropout_ratio=0.25, in_place=True)


```

```python
#output layers

    n.score_all = L.InnerProduct(n.drop1, num_output=n_classes * n_classifiers, weight_filler=dict(type='xavier'))
    n.score = L.Reshape(n.score_all, reshape_param={'shape': {'dim': [batch_size, n_classes, n_classifiers]}})

    if mode == 'train':
        n.loss = L.SoftmaxWithLoss(n.score, n.label)
    else:
        n.loss = L.Softmax(n.score)

```

To understand how all of these layers are connected, look more closely, and you can see that in each layer, except the input layer, the first argument we are receiving is the layer we defined just one line above! So the output of one layer is the input of the next! That's why we have so many lines connecting the circles to each other in those network images we looked at earlier. LINES!! CONNECTIONS!!! 

Sooo. Now let's have a look into the input layer and its parameters. This is where we define what kind of data our network can be expecting.


```python
#input layer

    n.data = L.Input(input_param={'shape': {'dim': [batch_size, 1, n_rows, n_cols]}})
    if mode == 'train':
        n.label = L.Input(input_param={'shape': {'dim': [batch_size, n_classifiers]}})
        
```
Remember for a second what type of data we are feeding into our net: Images. All an Image is in our case is a 3 dimensional matrix 1 x `n_rows` of pixels x `n_cols` of pixels . This is exactly corresponding to what we saw earlier in our Terminal outut (Network Topology) when running `./generate_network`. 

```
data (1, 1, 100, 100) 	# (batch size (1), image dimensions (1 x 100 x 100))
```

What does this tell us? It tells us that our image will be represented as 1 node per pixel in the first layer. That is all that is happening in the first layer.

So what about this `if mode == 'train' `? Remember when I was talking about how this function gets a `mode`. So it looks like, for the training net, we don't just want to pass the image itself, but also some information about this image to learn with (the label). This will become important later on for the output layers, so for now I will push it aside a little. 

For the hidden layers: My goal is for you to grasp the basic concept of what is happening here. Not that you understand the math behind it. If that is your goal though, look at the book I mentioned in the begining.

You might be wondering 'why so many hidden layers?' and 'is it important to have exacltly this line up?' Nope! This probably the most variable part of the program. Depending on the problem you want to solve you can shuffle these layers around, add more layers, add different layers, delete layers and so on. To give you an idea about what types of layers there are, have a quick look into the [Caffe Layer Catalogue](http://caffe.berkeleyvision.org/tutorial/layers.html). 

But to give you a basic idea about what our layers are for:

* Convolution → extracting information of parts of an image and passing this information to different parts of the image
* ReLu → what part of the image is important? for example where are my white parts of the image?
* Pool → downsize my image (compression)
* LRN → normalisation
* Innerproduct → gather all the information of all the image. And take all of it into consideration. (fully connected layer)
* Dropout → make sure your network is not becoming bias. 

Also, There is one important parameter that pops up a couple times in this part: 

* ` weight_fillers` -> these are the weights we are 'applying' to the values in the nodes, these are the weights the neural net will try to optimize/learn during all of the iterations. The important thing is that in the beginning they are randomly initialized. That's what all the `xavier`s are for. Check [this](https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/) out for more information.

OK, last layer section: 

```python
#output layers

    n.score_all = L.InnerProduct(n.drop1, num_output=n_classes * n_classifiers, weight_filler=dict(type='xavier'))
    n.score = L.Reshape(n.score_all, reshape_param={'shape': {'dim': [batch_size, n_classes, n_classifiers]}})

    if mode == 'train':
        n.loss = L.SoftmaxWithLoss(n.score, n.label)
    else:
        n.loss = L.Softmax(n.score)
```
Innerproduct we already talked about and Reshape is just for format purposes. SoftmaxWithLoss and Softmax are the important ones. They both return you the probablity of the input being a circle or square, the 'answer' in other words. The only difference is SoftMaxWithLoss also gets that extra information which we talked about in the input layers, the label, or desired outcome. What SoftmaxWithLoss then does is look at how far off it was to having the right answer and sending that back into the net (backpropagation). This information is needed to adjust those weights. THIS IS WHERE THE LEARNING IS HAPPENING ! 

In other words to make this more vivid and coming back to the baby example: 

If we have a little boy who's sister stole his toy. He will pull her hair (applying weights). To the parent, this is a wrong outcome. the desired outcome ('label') would have been to ask her nicely to give it back. The parent (SoftMaxWithLoss) will put the boy in timeout, for the boy to realize his actions were not correct and maybe think of what he should have done differently  (backpropagation). He will then adjust his behavior (apply different weights) the next time his sister steals his toy. Once the boy is grown and he has learned which weights to apply he no longer needs his parents to discipline him (SoftMax). 

Yes, yes I know this would leave me extremly unsatisfied, something that you are feeling right now probably. But just trust me that this is what's happening. And as I said if you are unsatisfied, check out the book or [here's](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/) also a good explanation

One thing which is interesting to look at is this:

![s_conv1](/examples/squares_conv1.png)
![c_conv1](/examples/circles_conv1.png)

Since we know that our `conv` layers are extracting information and have weights, that must mean it's learning what information it needs to extract. So all those two images at the top are showing us is the network's learned weights applied to two test images. 

Have a quick look at  `GetSolver()` yourself. I dont want to waste much time on this. All this really is is set a bunch of parameters, don't worry too much about them at this point. All we need to remember for later is that the solver has acces to the train net and the test net structures `train.proto` `test.proto`. 

Yay! We made it through the definitions! One thing to keep in mind these are only definitions of the layers, no actual computation has happened so far. We are still at level new born baby.



### Training the Network (train.py)

Again, for best results skim over the `train.py` file before continuing. 

As i explained before, in training, we are basically feeding the network with images, letting the network do it's thing, then each 100 iterations checking in and seeing how 'smart' it already is. Imagine it like letting the baby grow up and learn from his actions/correct it's actions and every year we just have a chat with the kid and figuring out what kind of morals it's building up.

There are a couple of things in the `train.py` file that I would like to take a closer look at and then we'll see if we can piece it all together. 

* First things first, this is were we have serious computations going on so it's important for you to know if you're using your gpu or your cpu. If you're not sure you're probably using a cpu. Depending on which one you are using uncomment/comment the correct line as follows:

``` python
# caffe.set_mode_cpu()
caffe.set_mode_gpu()
```

* don't worry about the `random_seed`.

* `algo.GetBatch` : Here we are creating a batch of images and their corresponding 'label'. if you go in the `algo.py` file you can check out the code for this yourself.  Some examples of what these images look like : 

![c](/examples/circ_1.png) 		![s](/examples/squ_1.png)

* `solver = caffe.get_solver('models/solver.proto')`: caffe's tool for us to load the neural net architecture from our protobuf file. Remember how when we were creating our solver we gave it the training net and the test net architecture? Well our `solver` now contains 2 nets one training and one test net based on our architectures : the training net (`solver.net`) and the testing net  (`solver.test_nets[0]`).

* Both these nets have `blobs`. [Blobs](http://caffe.berkeleyvision.org/tutorial/net_layer_blob.html) are literally just a weird way of saying layer. Don't ask me why just roll with it. Remember where we were building our network, we created all these layers like `n.data` and `n.label` or `n.loss`? See the similarity between those and `blobs['data']`, `blobs['label']`, `blobs['loss']`. That's because they're the SAME ONES! The differnce now is that we're not just defining the architecture but we are actually filling the layers with data or accessing the data in these layers. Like so: 

  * `solver.net.blobs['data'].data[...] = batch_images` → filling training net with images
  * `solver.net.blobs['label'].data[...] = batch_labels` → filling training net with corresponding labels
  * `solver.test_nets[0].blobs['data'].data[...] = batch_images` → filling test net with images
  * `solver.test_nets[0].blobs['loss'].data.argmax() == batch_labels[0,0]`  → accesing computed data to compare with labels
  
  
* `solver.step(1)` : Calling this function will take the batch of data we just filled the net with, send it forward through all the layers, compare the result the net comes up with to the `labels` we filled it with and correct the net's weights by sending the error back (backpropagation).

* `solver.test_nets[0].forward()` : Exactly what's happening in `solver.step(1)` but without the correction part. 

* `solver.net.save("models/weights.proto")` : Caffe's tool to save all those weights it has been correcting and learning up until now, to use them again later. Because what's the point of learning something if all you'r going to do is throw it all away right?
  
Now, if you take another look at `train.py` hopefully, what's happeninghere is making a little more sense. Load up our solver with a train and test net based on the architecture defined earlier, give it `n_iter` many batches of data to learn from. After every `test_interval`'th iteration run a test to see how many images it can already correclty classify. 

If you run `./train` again, the increasing accuracy should now makes sense. Pretty cool huh? :)  
     
### Classify with the trained Network (classify.py)

If you understood what's going on in the `train.py`, this should go pretty quick. Again, for best results skim over the `classify.py` file before continuing.


The only things that should be new is :
* `net = caffe.Net('models/deploy.proto', 'models/weights.proto', caffe.TEST)` : Again, all this is is a caffe tool to load up our deploy net (remember the deploy net is identical to the test net which makes sense if you think about it for a minute) and those weights we just trained. 

Other than that it should be pretty straight forward, For 1000 iterations, create a test image and its label, feed it through the network with the trained weights, compare the net's output to the label, count how many time it gets it correct and print out the accuracy! 

Also, there are two parts to this program depending on how you run it: `./classify.py` or `./classify.py 1` . The difference is if you're passing an extra argument ` 1`  or not. Here's the difference in code :

```
if len(sys.argv) == 1 and test_it % 10 == 0:
		print "\rIter:", test_it,

```
vs. 

```
if len(sys.argv) == 2:
		if net.blobs['loss'].data.argmax() == 0:
			print "Circle"
		else:
			print "Square"
		if algo.ShowImage(batch_images[0]) == 113:
			quit()

```


Go ahead and run both. As you can see the first one simply rushes through 1000 differnt images and give you the accuracy of all of them. The second way of running it you will see little images pop up and in you're terminal you can see the magic happen! It correctly guesses which it is :O.  Only know, you are smart and you know why this works or at least the concept of why it works. 

I really encourage you to not give up here. There are endless possibilities of what these neural nets are capable of. Play around with the layers a little bit or read up on some of the things that are happening! Hopefully this short turoial was able to spark your fascination a bit for deep learning. 

