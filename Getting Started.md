#Getting Started

###On OSX
There are a few ways to get iPython and the necessary dependencies installed. One option is to use the package manager [homebrew](http://brew.sh/). [This](http://www.lowindata.com/2013/installing-scientific-python-on-mac-os-x/) guide has helped me in the past, but is a bit dated.

Another great option, and probably the simplest is to use the [anaconda](https://store.continuum.io/cshop/anaconda/) or miniconda distrubtions from endthought. 

For some great context, check out [this](http://youtu.be/FgfKA-HJFI0) talk from Travis Oliphant, the creator of numpy and the anaconda distribution.

Finally, a very thorough [guide](http://sourabhbajaj.com/mac-setup/) to getting going on OSX.

###On Windows
Install Linux or buy a MacBook. 

Just kidding, there's some way to make this work, but I have no idea what it is.

##Running iPython Notebook
Once all software is installed, navigate the the proper directory in the terminal and type:

<code>ipython notebook --pylab inline</code>

The pylab option automatically imports useful things like scipy, numpy, matplotlib. The inline command makes the plots inline with the notebook, as opposed to popping them out. 

##Working with audio
A great audio tool for python is [scikits.audiolab](https://pypi.python.org/pypi/scikits.audiolab/). You can install scikits audiolab with pip: 

<code>pip install scikits.audiolab</code>

You may need to install [libsndfile](http://www.mega-nerd.com/libsndfile/) for audiolab to work properly.

##Editing
When not editing in the notebook, I recommend [Sublime Text](http://www.sublimetext.com/).

##Github
[Nice summary of fundamentals](https://www.youtube.com/watch?v=0fKg7e37bQE). Generally on this project, contributors will fork the repository to make a local copy, edit away, and submit pull requests to merge thier code back in. 

To keep the repository nice and clean, you can prevent your machine from pushing up files that aren't needed in the repository (.pyc, compiled python scripts, and on OSX .DS_Store system files). [Here](http://devoh.com/blog/2013/01/global-gitignore)'s a nice guide to setting this up on osx: 

Your gitignore file should contain: 
```
#Ignore compiled python scripts:
*.pyc

#Ignore OSX Files
.DS_Store

#Ignore python checkpoints:
*.ipynb_checkpoints
```


