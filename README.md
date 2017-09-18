# Project Progress Journal

<!-- TOC -->

- [Getting Started](#getting-started)
    - [Major Happenings](#major-happenings)
    - [Roadblocks](#roadblocks)
    - [Questions](#questions)
- [Week 1](#week-1)
    - [Major Happenings](#major-happenings-1)
    - [Roadblocks](#roadblocks-1)
    - [Prospective](#prospective)
- [Week 2](#week-2)
    - [Major Happenings](#major-happenings-2)
    - [Roadblocks](#roadblocks-2)
    - [Prospective](#prospective-1)
- [Week 3](#week-3)
    - [Major Happenings](#major-happenings-3)
    - [Roadblocks](#roadblocks-3)
    - [Prospective](#prospective-2)
- [Week 4](#week-4)
    - [Major Happenings](#major-happenings-4)
    - [Roadblocks](#roadblocks-4)
    - [Prospective](#prospective-3)

<!-- /TOC -->

## Getting Started

### Major Happenings

- Installed [Visual Studio Code](https://code.visualstudio.com/download) as my main code editor after having issues with python in Atom.
- Installed [tensorflow](https://www.tensorflow.org/install/) on my home laptop - this should be easier to use than Theano.
- Installed [bash on ubuntu on windows](https://msdn.microsoft.com/en-us/commandline/wsl/install_guide) because I didn't want to deal with Powershell.
- Broke my previous code for the Terry project into separate modules, to make reading and maintaining it easier.
- Set up [Codebeat](https://codebeat.co/) to analyze my code and suggest ways to improve it.
- After a lot of searching around, finally settled on a [tutorial](https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767) for creating RNNs in tensorflow.
- Followed the tutorial through Part 3. At this point, I have:
    - An RNN that succesfully trains on pre-formatted text datasets (i.e., the training loss goes down with time).
    - The RNN uses the built-in API such that I have very little math code in my work.
    - The LSTM cells from the tutorial are replaced with GRU cells, for they are simpler, require less computations, and apparently produce very similar results.

### Roadblocks

- I failed to get the GPU version of tensorflow to run on my laptop.
- Documentation for tensorflow can be confusing, and is not as extensive as the documentation for Theano.
- Unlike with Theano, I could not find a tensorflow tutorial that showed me how to do exactly what I needed it to (although, there are plenty of general RNN tutorials in tensorflow).
- Apparently, my tensorflow version wasn't compiled with certain options that could speed up training on my laptop.

### Questions

- How do I 'publish' my python modules such that I will be able to re-use them in the main project? At the moment, I'm thinking either publishing them via 'pip', or creating a git submodule out of them.

## Week 1

### Major Happenings

- Worked on getting my RNN to generate text.
- Gained a better understanding of how the RNN API in tensorflow works.
- Finally fully understood how the 'randomness' in numpy worked with the text generation.

### Roadblocks

- The RNN is generating duplicate gibberish, with text samples containing phrases like "I I I I I I I I I were here were were..."
- As usual, debugging a running RNN is difficult, although so far the tensorflow errors have been much easier to read than the Theano ones.
- When attempting to have the RNN learn where to put spaces, the RNN never once output a space token, despite it being the most common token in both the training and output data.
- It seems that the RNN API does not convert data to one-hot vectors automatically, like I thought it would. I may have to do that step manually. The good news is that this step may help with my text generation problem.

### Prospective

- Is learning to use the tensorboard feature going to take a long time? And would it help me diagnose problems earlier?
- For the main Kindschi Fellowship project, how do we judge the success of the network? We can't really test it in a real-life scenario, so we might not actually know how useful this is.
- How do we deal with the network generalizing movement patterns? We will either have to manually give it 'seeded' movements to represent the first couple steps, or find a way to group the training data and have an extra 'feature' representing the data group. Grouping the data, however, simply based on location, is a task that would seem to require a separate neural network, and we may not have the time to design one.

## Week 2

### Major Happenings

- Fixed an error where one-hot vector encodings weren't happening where I thought they were.
- Added an embedding layer to the RNN.
- The above two points helped to more-or-less solve the text generation problem. This still has to be tested on a larger dataset whenever a computer with a GPU becomes available.

### Roadblocks

- Using one-hot vectors with the tensorflow API without adding dimensions to tensors proved challenging. Luckily, with the embedding layer, the one-hot vectors were simulated instead of actually implemented.

### Prospective

- Need to find a way to make the RNN code less coupled from its current implementation, so that it could be readily used for the new dataset. 

## Week 3

### Major Happenings

- Installed [ray](http://ray.readthedocs.io/en/latest/install-on-ubuntu.html) on my laptop to enable saving of tensorflow variable weights.
- Added functionality to save and load weights from a trained model.
- Installed [tensorflow-tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) to enable visualization of the tensorflow graph, as well as variable summaries.
- Added functionality for tensorboard to display the tensorboard graph, and some variable summaries.
- Added an accuracy tensorflow operation to measure the accuracy of the model (as opposed to just the cross-entropy loss, which was the case before this change).

### Roadblocks

- I first attempted to save and load weights with the `tensorflow.Saver()` object, but kept running into an error where, after loading the previously saved graph, all previously saved changes would be lost during training.
- I could not solve the above problem, and ended up using a third-party tool (ray) for saving and loading tensorflow weights. 
- I had the wrong tensorflow version for use with tensorboard, which caused errors when trying to visualize variable summaries until I upgraded tensorflow to version 1.3.0.
- I had to install `tensorflow-tensorboard` using pip, which shouldn't have been the case according to tensorflow's docs. This may have been fixed after upgrading tensorflow, but I'm not keen on fixing what isn't broken.

### Prospective

- The RNN model should now be ready for training on a larger dataset - waiting on `Seawolf` in order to do that.
- In the mean time, the model can be further improved with more layers, a YAML config file for storing settings (as opposed to command-line arguments, since there are so many configurable options), and making it more general-purpose (currently a lot of minor configuration changes would break it).

## Week 4

### Major Happenings

- Made datasets more regular (same start and end tokens as opposed to different ones for every type of dataset, etc.)
- Added functionality to create character-level datasets.
- Added functionality to use YAML config files instead of passing in all the command-line arguments.
  - Config files are stored on disk, which allows for easy re-use, and makes it easy to check what configuration was used when disseminating the fellowship results.

### Roadblocks

- Character-level datasets ended up taking much more space than I thought they would. I had to reduce the number of stories used from over 8000 to approximately 7500 for them to be able to be hosted on GitHub.
- I'm not entirely sure that the RNN can train on these new datasets yet - will have to make sure of that in the next week.

### Prospective

- I don't have a way to read non-csv data files yet. I'll have to implement that when I find out what kind of dataset I'll be using.
- There's the problem of seemingly concurrent training when more than one RNNModel object is present in the same script. I will have to find a work-around for that. This is so that each 'run' will contain a certain number of epochs, after which the model saves its weights. It's either that, or create a more complex Saver object that will take care of the 'run' management.
  - Currently, the epochs are being ignored altogether, and the model only saves its weights after it has completed all the epochs that were requested.
- I'm not sure that my dataset can handle integer values correctly yet. Just like in the character dataset, there will be no need for an embedding layer in the integer dataset.