#Getting Started

##Python Stack
Several options exist for running scripts on your machine. On OSX, I recommend using the package manager [homebrew](http://brew.sh/).

Building a nice functonal version of python with all needed dependencies can be tough, especially if you're coming from a language like MATLAB. You will learn a bit about how software and computers work though!

###On OSX
There are a few ways to get iPython and the necessary dependencies installed. One option is to use the package manager homebrew. This guide has helped me in the past, but is a bit dated: http://www.lowindata.com/2013/installing-scientific-python-on-mac-os-x/

Another great option, and probably the simplest is to use the anaconda or miniconda distrubtions from endthought: 
https://store.continuum.io/cshop/anaconda/

For some great context from Travis Oliphant, tha creator of numpy and the anaconda distribution:
http://youtu.be/FgfKA-HJFI0

Finally, a very thorough guide to getting going on OSX:
http://sourabhbajaj.com/mac-setup/

###On Windows
Install Linux or buy a MacBook. 

Just kidding, there's some way to make this work, but I have no idea what it is.

##Running iPython Notebook
Once all software is installed, navigate the the proper directory in the terminal and type:

<code>ipython notebook --pylab inline</code>

The pylab automatically imports useful things like scipy, numpy, matplotlib. The inline command makes the plots inline with the notebook, as opposed to popping them out. 

##Working with audio
A great audio tool for python is scikits.audiolab:


##Editing
When not editing in the notebook, I recommend Sublime Text: http://www.sublimetext.com/

##Github
Nice summary of fundamentals: https://www.youtube.com/watch?v=0fKg7e37bQE Generally on this project, contributors will fork the repository to make a local copy, edit away, and submit pull requests to merge thier code back in. 

To keep the repository nice and clean, you can prevent your machine from pushing up files that aren't needed in the repository (.pyc, compiled python scripts, and on OSX .DS_Store system files). Here's a nice guide to setting this up on osx: http://devoh.com/blog/2013/01/global-gitignore

Your gitignore file should contain: 
```
#Ignore compiled python scripts:
*.pyc

#Ignore OSX Files
.DS_Store
```


