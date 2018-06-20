
Functional Imaging Compression, Denoising and Demixing
======================================================

We introduce a pipeline to compress, denoise, and demix* several types of functional imaging recordings including:
    
    * Calcium Imaging (1p, 2p, 3p)
    * Voltage Imaging (1p)
    * Widefield


Installation
------------

After installing the dependencies, install from the master branch as:


.. code-block:: shell

    git clone https://github.com/paninski-lab/FunImag.git funimag
    cd funimag
    pip install -e .
    
Note: We currently only support Linux OS (tested on Ubuntu 16.04 and 18.04).


Demos
-----


*   Denoising and demixing demos with a sample data from [CaImAn](https://github.com/flatironinstitute/CaImAn).
        
        * [Denoising demo]()
        * [Demixing demo]()
   

Dependencies
------------

*   [TREFIDE](http://github.com/ikinsella/trefide)


Running tests
-------------

In progress


Reference
---------

If you use this code please cite the paper:

E. Kelly Buchanan, Ian Kinsella, Ding Zhou, Rong Zhu, Pengcheng Zhou, Felipe Gerhard, John Ferrante, Ying Ma, Sharon Kim, Mohammed Shaik, Yajie Liang, Rongwen Lu, Jacob Reimer, Paul Fahey, Taliah Muhammad, Graham Dempsey, Elizabeth Hillman, Na Ji, Andreas Tolias, Liam Paninski
bioRxiv 334706; doi: https://doi.org/10.1101/334706 
