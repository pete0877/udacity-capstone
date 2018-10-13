# udacity-capstone

This is my Udacity Machine Learning Nano Degree capstone project.

The aim of the project is to develop ML method that is able to 
identify video frames (in videos of American football games) 
that contain line of scrimmage. The frames that contain line of 
scrimmage indicate beginning of a play / action.

See the `proposal.pdf` file for more details.

The `report.pdf` for the final project report. 

#### To run the project:

1. Create a Python 3.5+ virutal envoronment and activate it. 
1. Install required packages by running:

    ```pip install -r requirements.txt```

    Note that the `av` package requires that you have the FFMPG
    library / tools installed. The `av` package is only needed if you 
    intend to run some of the frame-extracting scripts. Running these 
    scripts is not necessary as all the images have already 
    been extracted and included in the `./data-images` folder. 
    If you intend to run frame-extracting , please uncomment the 
    ```# av==0.4.1``` line prior to running the command above.
1. Additionally, if you indent to run some of the frame-extracting scripts
   you will need to download the video media files that could not be
   included as part of this GitHub repository (due to their size). Again, 
   this step is **not necessary** since the frames have already been extracted. 
   But if you want to run this step yourself, download all the files from this 
   DropBox location into the `./data-videos` folder first:
   
   https://www.dropbox.com/sh/9t48voatzxy85oq/AAB5jZuJThaf5q2XYt6l1eNaa?dl=0
   
   Then run `python extract_frames.py`.
1. To run the project itself, open the `transfer_learning.ipynb` in a Notebook.  

Final project report has been included in report.docx and report.pdf.