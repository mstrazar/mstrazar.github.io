---
layout: post
title:  "Reduce, reuse, recycle t-SNE projections "
date:   2021-08-24 20:00:30 +0100
author: "Martin Stražar"
categories: orange image-analysis deep-learning 
---

<p>Exploratory data analysis and modelling projects often start with a compact visualization of available data.
Staples like principal component analysis (PCA) or multidimensional scaling (MDS) methods achieve this by
finding a two dimensional summary of the data that preserves the main axes of variation. They reveal potential
patterns and (if we are lucky) some structure in a multi-dimensional data set.</p>

<p> The <a href="https://distill.pub/2016/misread-tsne/">t-distributed Stochastic neighbour Embedding (t-SNE)</a> 
is a modern variation on these methods, 
that more closely resembles to how a human user might interpret such 
figures. Essentially, we only care that very similar points appear close together to a given point, 
while more distant points do not influence its positioning. This allows dramatically more appealing 
projections, that although harder to quantify, tend to produce clumped, grape-like clusters. </p>

Few fields have used t-SNE to such extent as single-cell transcriptomics, where it provides a natural tool 
for revealing different cell types or states.

The non-linear and data-driven nature of t-SNE makes it hard to re-use and compare between different data sets.
In our latest methods work, led by <a href="https://github.com/pavlin-policar">Pavlin G. Poličar</a>, 
we develop a principled way to perform t-SNE with two different 
data set. One data set is termed the <i>primary</i> or the <i>reference</i> data set, where t-SNE is computed as usual.
When bringing in a second data sets, we implement an efficient and parallel way to find the best position for each point 
in the reference projection.

Image the reference projection as a scaffold that establishes possible positions for each point. This way, large
reference t-SNE projections of millions of cells can be used to make sense of new, smaller data sets 
that are generated!    

Read more in <a href="https://link.springer.com/article/10.1007/s10994-021-06043-1">Machine learning</a>.
<br/>

<figure>
<img width="500" src="/img/posts/tsne-embedding/example.jpg"/> 
<br/>
<figcaption> The t-SNE embedding method implements a projection of new samples into an existing, 
reference t-SNE projection.
</figcaption>
</figure>



