--------------------------------------------------------------------------------------
Item and User Based Recommender System for Ml-100k ratings
Programmers/Researchers: Paul Choi, Base Elzatahry, Rida Shahid, Natnael Mulat
Institution: Davidson College
--------------------------------------------------------------------------------------
Data - 'ML-100k'
======================================================================================
This program is designed to ran item-based and user-based recommender system for 
MovieLens 100K movie ratings with 943 users on 1682 movies, where each user has rated 
at least 20 movies.The data was collected through the MovieLens web site
(movielens.umn.edu)

Libraries Used
======================================================================================

   NumPy	       -- a library for the Python programming language, a large collection 
		of high-level mathematical functions to operate on these arrays		   
   Matplotlib   -- a plotting library for the Python programming language and its 
		numerical mathematics extension NumPy.
   Pickle       -- a library for serializing and de-serializing a Python object 
		structure. Any object in Python can be pickled so that it can be 
		saved on disk. 	


DETAILED DESCRIPTIONS OF COMMANDS
======================================================================================
RML   --  To read the ml-100k data into the program, please note that this command
	    is a required command for other function calls
P     --  To print the users and items in an orderly fashion ("user item") for 
	    verification and visualization purposes

V     --  To validate some of the users and their ratings for specific movies, note 
	  that this command already has specified user to print, please read comment 
	  in program

S     --  To print useful stats from the data, specifically:
		Number of users
		Number of items
		Number of ratings
		Overall average rating
		Average item rating
		Average user rating
		User-Item Matrix Sparsity
		Popular items -- most rated
		Popular items -- highest rated		
		Overall best rated items (number of ratings>=5)

D      --  To get the Euclidean distance similarity for users 

P      --  To get the Pearson Correlation for users 

SIMU   --  reads/writes a User-User similarity matrix, please note that this has 
	  additional command when run that specifies the similarity method

Sim    --  reads/writes a Item-Item similarity matrix, please note that this has 
	  additional command when run that specifies the similarity method
 
LCVSIM --  Leave one out cross validation that returns the error and error list,
	   please note that this command will ask for choice of recommender system
	   between user-based and item-based
 
-----------------------------------------------------------------------------------------
SET OF COMMANDS TO GET RESULTS 
-----------------------------------------------------------------------------------------
RML --> p 

Prints the user and items from the data that is read

RML --> V 

Prints some of the data that is read and to see if it matches the records of ml-100k

RML --> S 

Prints the stats for the data that the program read, please refer to S command for more
 detail on type of stats it prints

RML --> D 

Prints the Euclidean distance similarity between each users from the data the program 
 Read

RML --> P 

Prints the Pearson Correlation between each user from the data the program 
 Read

RML --> SIMU --> LCVSIM --> U

This set of command gives us the MSE, MSE_list, MAE, MAE_list, RMSE, RMSE_list for
 a user based recommendation. This set of command uses the similarity matrix for user-
 user to make predictions. The leave-one-out-cross validation uses the matrix and 
 recommender functions to find the prediction and actual value for calculation of errors. 

RML --> SIM --> LCVSIM --> I

This set of command gives us the the MSE, MSE_list, MAE, MAE_list, RMSE, RMSE_list for
 a item based recommendation. This set of command uses the similarity matrix for Item-
 Item to make predictions. The leave-one-out-cross validation uses the matrix and 
 recommender functions to find the prediction and actual value for calculation of errors. 

ACKNOWLEDGEMENTS
============================================================================================

Thanks to DR. CARLOS E.SEMINARIO for all the guidance he provided through out this research.