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
- [Week 5](#week-5)
    - [Major Happenings](#major-happenings-5)
    - [Roadblocks](#roadblocks-5)
    - [Prospective](#prospective-4)
- [Week 6](#week-6)
    - [Major Happenings](#major-happenings-6)
    - [Roadblocks](#roadblocks-6)
    - [Prospective](#prospective-5)
- [Week 7](#week-7)
    - [Major Happenings](#major-happenings-7)
    - [Roadblocks](#roadblocks-7)
    - [Prospective](#prospective-6)
    - [Resources](#resources)
- [Week 8](#week-8)
    - [Major Happenings](#major-happenings-8)
    - [Roadblocks](#roadblocks-8)
    - [Prospective](#prospective-7)
    - [Resources](#resources-1)
- [Week 9](#week-9)
    - [Major Happenings](#major-happenings-9)
    - [Roadblocks](#roadblocks-9)
    - [Prospective](#prospective-8)
- [Week 10](#week-10)
    - [Major Happenings](#major-happenings-10)
    - [Roadblocks](#roadblocks-10)
    - [Prospective](#prospective-9)
    - [Resources](#resources-2)
- [Week 11 (October 30 - November 5)](#week-11-october-30---november-5)
    - [Major Happenings](#major-happenings-11)
    - [Roadblocks](#roadblocks-11)
    - [Prospective](#prospective-10)
- [Week 12 (November 6 - November 12)](#week-12-november-6---november-12)
    - [Major Happenings](#major-happenings-12)
    - [Roadblocks](#roadblocks-12)
    - [Prospective](#prospective-11)
- [Week 13 (November 13 - November 19)](#week-13-november-13---november-19)
    - [Major Happenings](#major-happenings-13)
    - [Roadblocks](#roadblocks-13)
    - [Prospective](#prospective-12)
- [Week 14 (November 20 - November 26)](#week-14-november-20---november-26)
    - [Major Happenings](#major-happenings-14)
    - [Roadblocks](#roadblocks-14)
    - [Prospective](#prospective-13)

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

## Week 5

### Major Happenings

- Added a shell script that uses config files to generate a bunch of text datasets.
- Added more meta-info to the saved datasets, for easier saving/loading inside a model.
- Code refactoring.
- Attempted to use new datasets to generate sentences.

### Roadblocks

- Generated sentences with the new dataset tend to be biased towards UNKNOWN_TOKENS, repeat words.
- When looking at the characters in my character-level datasets, I see a lot of characters in the 'extended ASCII codes' category. This suggests I either have some garbage input, some input in non-English languages, or a lot of input with weird formatting issues, which needs to somehow be cleaned up.
- I noticed that the way I create batches in my datasets doesn't work very well. They lose the essence/meaning of the 'start' and 'end' tokens, since the batches no longer start and end with them.

### Prospective

- I need to find a way to clean the training data without getting rid of a large chunk of my data.
- Creating batches will cause more overhead, and more 'meaningless' data because the batches will have to be padded.
- There is a minor dilemma in choosing which token to use to batch the data. Currently, I am using the 'end token', although a case can be made for creating another special token used exclusively for batches.
- The data for my Kindschi research itself will be available soon.

## Week 6

### Major Happenings

- Created a batchmaker file that converts training data into batches.
- Got a look at sample data for my research. It seems straightforward, but will need to be re-formatted for it to work with the model.
- Got more ideas on how to measure the accuracy of the network.

### Roadblocks

- The batchmaker does not work with the network yet - it'll have to be adjusted to work with the new batches.

### Prospective

- Re-formatting the data could take a while. There's also the question of the time-steps to take into account. Do I look at the data in terms of days, in terms of hours, or just in terms of number of movements?
- I might want to change the RNN type I use to a dynamic rnn, to allow for batches with different sizes.

## Week 7

### Major Happenings

- The RNN now trains with the new batches.
- May have figured out a simple way to ignore 'unknown' tokens in the input - either exclude them in the argmax operation, or, if randomizing the output, do a softmax on the probabilities that don't include the unknown token (which will be easy because the unknown token is always last).
- Wrote tests in `pytest` for the batchmaker, because it's all python (no tensorflow stuff), and its errors could be hard to catch later on.

### Roadblocks

- Tensorflow seems to sometimes treat lists as a single object, rather an as an array. Had to convert batches to numpy arrays to get it working.
- Training seems slower with the new batches. Can't tell if it's the batches themselves, or all the padding.
- Trying to use the dynamic rnn throws out an error on input tensor shape, even though it seems to be the correct shape. (It complains that the tensor is of rank 2, when it is actually of rank 3).

### Prospective

- Operations on tensors with more than two dimensions are hard to visualize, which slows down my understanding of what the network is doing. I might be worth it to do some practive with 3d and 4d tensor operations by hand.
- It might be worthwhile to benchmark the RNN using the GPU on seawolf to find the parameters that make training fastest, and then use those from here on out.
- I should make the RNN code into a standalone repo, so that I can reuse it in other projects.

### Resources

- Paper on neural responding machines with RNNs: https://arxiv.org/abs/1503.02364

## Week 8

### Major Happenings

- Got the dynamic RNN api to work by moving a call to `tf.unstack`.
- Ran multiple benchmarking scripts to find the optimal batch size and sequence length for optimal training time.
    - Training time with respect to both batch size and sequence length appears to be parabolic in shape.
        - So far, the optimal batch size was 500 (in a dataset with 1000 examples)
        - The optimal sequence length was between 10 and 15
    - Fastest training time was about 1:14 minutes for 1000 examples for 100 epochs (or, ~100000 examples for 1 epoch).
- Started on a script that converts the synthetic data into a sequence dataset.
    - Can probably make 3 kinds of sequences out of it:
        - Divide day into `x` chunks. Increment chunk time, keep same location if no hope made, else append new location.
        - Ignore time altogether, only represents hops made.
        - Try to make the network guess the location AND timing and number of hops.
- Fixed an error in batch creation where the batch isn't vertically padded (when there aren't enough examples to reach size of batch).

### Roadblocks

- Because tmux works in the same directory as my code, I caused my benchmark scripts to fail multiple times by editing the RNN code while the benchmark script was running.
    - Will probably be fixed if the RNN code is made into a submodule.

### Prospective

- Need to tweak my accuracy-producing code.
- Code may be further refactored to make the model simpler and more flexible.
- A genetic algorithm could be used to select the optimal hyperparameters to train the model.

### Resources

- Excerpt of book that includes a high level overview of genetic algorithms: https://goo.gl/KjLRLW
- A blog with a tutorial on genetic algorithms: https://goo.gl/jHft7W

## Week 9

### Major Happenings

- Moved the RNN code into its own repo on github
- Created 4 trainable datasets from the synthetic data
    - 200, 10000, 100000, and 318256 examples
- Added multiple RNN layers
- Added dropout after each layer

### Roadblocks

- On smaller datasets, the RNN seems to have trouble learning hops (timesteps it displays are all the same)
- Using a genetic algorithm could prove to be more difficult than I had thought

### Prospective

- Need to partition data into training, validation and testing partitions.
    - Can use cross-validation to make up for not having a large enough dataset.
- The genetic algorithm should be pushed back until I get the accuracy code implemented.
- Need to get the accuracy implemented soon so I can start collecting data.

## Week 10

### Major Happenings

- Added code to break the dataset down into training, validation and testing partitions
    - 10% of entire dataset goes into the testing partition
        - The testing partition is used after training as a definitive performance measure
    - Rest of dataset broken down into 10 partitions for cross-validation
- Tweaked RNN code to make cross-validation work

### Roadblocks

- Spent way too much time trying to track down and fix and off-by-one error
- The testing partition isn't currently being used
- Shuffling a large enough dataset may end up being computationally costly (done to prevent the test partition from inheriting the ordering from the dataset)
- There are now essentially 10 epochs for every actual epoch (since training goes over 9/10th of the non-testing data 10 times in every epoch)

### Prospective

- It may be prudent to feed the predictions and labels into separate placeholders when calculating the accuracy. This way, cross-validation won't cause weird tensorboard output.
    - This would also mean that these calculations can be done on the GPU instead of on the CPU, which would save time (probably).
- Still need to calculate accuracy...

### Resources

- Paper on early stopping mechanishms: https://goo.gl/eUCQ9Z

## Week 11 (October 30 - November 5)

### Major Happenings

- Worked on logging model performance epoch, instead of by minibatch.
    - Works by aggregating logits and labels for each minibatch, and then feeding those into the loss and accuracy calculation ops.
- Refactored code for training and creating batches.
- Added metadata to batches to indicate whether the batch contains data that is the beginning and/or ending of a sequence.

### Roadblocks

- Project is getting pretty big, finding what needs changing is getting trickier.
- The whole operation seems pretty inefficient.
- I'm getting a pretty convoluted set of tensorflow ops that's hard to read.

### Prospective

- Some tensorflow tests would be great to make sure the model works as expected.

## Week 12 (November 6 - November 12)

### Major Happenings

- Changed to a more efficient way for calculating performance.
    - Using an aggregator class, I keep track of average performance data for each minibatch
    - The averages are then combined for each incoming minibatch
- Updated some tests to reflect latest batch implementation changes.
- The model now evaluates performance on the test partition.
- Added accuracy calculations using a mask.
    - Loss calculations were refactored to use mask as well.
- Added pyplot graphs for the performance data, because they're easier to copy/paste and view than tensorboard (and they don't have a problem with un-synched timesteps)
- Started first run on full dataset.

### Roadblocks

- Turns out the operations from before WERE inefficient: both Python and the GPU ran out of memory while performing the performance evaluation.
- Somewhere down the line, the code for loading and saving the model weights broke.
- Tensorboard doesn't like data with un-synched timesteps, so I had to insert an extra epoch at the end to record performance on the test partition.
- There appears to be a problem with tensorflow on the GPU - training periodically breaks and the entire model has to be restarted.
    - Error received:
        ```
        E tensorflow/stream_executor/cuda/cuda_event.cc:49] Error polling for event status: failed to query event: CUDA_ERROR_LAUNCH_FAILED
        F tensorflow/core/common_runtime/gpu/gpu_event_mgr.cc:203] Unexpected Event status: 1
        Aborted (core dumped)
        ```
    - Also got segmentation faults a couple of times

### Prospective

- If I get saving and loading to work, then tensorflow crashing won't be as big a problem.
    - Model would have to be saved off after every epoch, not just at the end of training.
- Because of cross-validation, the training and validation losses don't differ much.
- Early stopping would be nice - this way I can train the model until it reaches it's best accuracy, rather than until some arbitrary epoch.
- There's got to be a better way at moving the performance data around the training methods.

## Week 13 (November 13 - November 19)

### Major Happenings

- Got first training results on the synthetic dataset:
    - Final accuracy: 77.22%
    - Accuracy highest during the night, lowest in the late afternoon / early evening
- Added a log-level option that allows seeing debug logs (I had forgotten to implement that earlier)
- Aggregated performance data now stores data for every epoch, which means it takes up more space (but not nearly as much as the inefficient method from a couple weeks ago).
- Implemented saving.
    - Added more metadata, including aggregated performance data for continuity.
    - Save after every epoch, and at best epoch (depending on accuracy).
- Performance data now created in 'root' training method, and passed down to the nested functions.
- Created a simple dataset that uses a 1:2:7 partitioning instead of cross-validation.
- Implemented a simple early stopping mechanism.
- Started a new run on full dataset using latest model.

### Roadblocks

- Figuring out a 'nice' structure for saving the model proved to be harder than I thought.
- Python's inheritance strategies are slightly different from what I was used to, which threw me off for a while.
- Had to restart the model multiple times because tensorflow crashed. I was actually surprised when the model finished training.
- The simple early stopping mechanism isn't working as well as I had initially thought.

### Prospective

- A python/bash script that automatically restarts the model when it crashes would be amazing.
- Perhaps some extra data in the network will help the model predict locations better? Example:
    - Home locations
    - Abstract movement patterns
    - Trip purpose
    - 'Person' ids
- On the other hand, maybe if I had intermediate call data, instead of just the ones that involved a change of location, I'd get better accuracies.
- Making a deeper network may be another solution, but might require extra techniques like neural highways.
- Using an attention layer may be yet another solution.

## Week 14 (November 20 - November 26)

### Major Happenings

- Finally signed that NDA, so I can use the real D4D data.
- Generated datasets that include home location as an additional feature.
- Created an abstract model class `RNNBase` that can be extended to give additional features.
- Created a model class `MultiInputRNN` that works with multiple tokenized inputs.
- Started a new run on the dataset with home locations using the new model.

### Roadblocks

- Model from last week stopped training without my notice.
- RNN with multiple inputs training much slower than with a single input (in terms of the drop in loss, not the time per epoch).
    - Simply an effect of multiple inputs?
    - An effect of a larger hidden layer?
    - Bad network parameters?

### Prospective

- Looks like I was doing comments wrong, I need to make params and return values into markdown lists for them to display correctly.
- Might be a good idea to refactor my dataset-producing code into a class or something, to make it easier to create new datasets from it.
- That genetic algorithms could be very useful indeed here for choosing the network parameters, but there may not be enough time to make one.