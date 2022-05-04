# FDDL
FDDL is a cooperative cache approach.  
This project contains the admission and eviction algorithms of FDDL.  


To run the project, you should prepare the dataset at first.

1. Prepare the original dataset and save it in `.csv` file:  
   The original dataset structure is as follows:  
   >#timestamp, contentID, contentSize  
   >1,5774,1  
   >10,11553,2  
   >23,3238,3  
   >30,46,1  
   >32,420,5  
   >...  

2. Extract features from original dataset:  
   The dataset structure is as follows:
   >  #timestamp, contentID, contentSize, frequency, interval between two same content (timestamp), interval between two same content (count), short term frequency, middle term frequency, long term frequency  
   >  1,5774,1,1,1,1,1,1,1  
   >  10,11553,2,1,10,2,1,1,1  
   >  23,3238,3,1,23,3,1,1,1  
   >  30,46,1,1,30,4,1,1,1  
   >  32,420,5,1,32,5,1,1,1  
   >  ...  

3. run `gateway.py` with `python gateway.py agent_index dataset_select cache_size`  
