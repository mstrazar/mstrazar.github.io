---
layout: post
title:  "Image analytics for the masses"
date:   2019-10-19 20:00:30 +0100
author: "Martin Stražar"
categories: orange image-analysis deep-learning 
---

<p>Computational image analysis got an immense boon with the advent of deep learning. Such technologies,
used by just about every social network, security camera system or self-driving car, are not as "plug-and-play" as we 
might like.</p>

<p>Training of these neural network is not only an expensive process, hungry for modern hardware, but also requires
substantial machine learning intuition and expertise. Fortunately, certain aspects of these models translate well 
between different problem domains. For example, pattern detectors trained on separating male and female faces could be re-used 
to train a model for a new task, perhaps linking a face to its identity.</p>

<p>Our work, led by Primož Godec and Matjaž Pančur utilizes this principle to bring deep learning for images closer 
to non-programmers. This is achieved by two main parts; first, we expose a large general-purpose neural network, ImageNet,
to compute numerical representation of input images. This is expected to work well, since ImageNet was trained to 
predict classes for a very general set of images, ranging from animals to furniture.</p>   

<p>Secondly, this process is implemented in <a href="http://orange.biolab.si/">Orange</a>, an established data visualization and machine learning toolbox
that allows us to utilize these techniques without coding. The package is very simple and one can use it e.g. to organize
a lingering collection of photos that has accumulated over the years. Or train a security camera to stalk 
your cat across restricted areas in the kitchen. </p>  
  
<p>Read more in <a href="https://www.nature.com/articles/s41467-019-12397-x">Nature Communications</a>.</p>

<br/>

<figure>
<img width="800" src="/img/posts/image-analytics/orange_image_example.jpg"/> 
<br/>
<figcaption> An Orange workflow that loads a set of images, computes their compressed representation (embedding)
and organizes them into groups as shown on the clustering tree. 
</figcaption>
</figure>

