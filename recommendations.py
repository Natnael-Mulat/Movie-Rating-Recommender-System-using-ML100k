'''
CSC381: Building a simple Recommender System

The final code package is a collaborative programming effort between the
CSC381 student(s) named below, the class instructor (Carlos Seminario), and
source code from Programming Collective Intelligence, Segaran 2007.
This code is for academic use/purposes only.

CSC381 Programmers/Researchers: << Paul Choi, Base Elzatahry, Rida Shahid, Natnael Mulat>>

'''

import os
from math import sqrt 
import numpy as np
import matplotlib.pyplot as plt
import pickle
# from scipy.stats import spearmanr
 



def from_file_to_dict(path, datafile, itemfile):
    ''' Load user-item matrix from specified file 
        
        Parameters:
        -- path: directory path to datafile and itemfile
        -- datafile: delimited file containing userid, itemid, rating
        -- itemfile: delimited file that maps itemid to item name
        
        Returns:
        -- prefs: a nested dictionary containing item ratings for each user
    
    '''
    
    # Get movie titles, place into movies dictionary indexed by itemID
    movies={}
    try:
        with open (path + '/' + itemfile, encoding='iso8859') as myfile: 
            # this encoding is required for some datasets: encoding='iso8859'
            for line in myfile:
                (id,title)=line.split('|')[0:2]
                movies[id]=title.strip()
    
    # Error processing
    except UnicodeDecodeError as ex:
        print (ex)
        print (len(movies), line, id, title)
        return {}
    except Exception as ex:
        print (ex)
        print (len(movies))
        return {}
    
    # Load data into a nested dictionary
    prefs={}
    for line in open(path+'/'+ datafile):
        #print(line, line.split('\t')) #debug
        (user,movieid,rating,ts)=line.split('\t')
        user = user.strip() # remove spaces
        movieid = movieid.strip() # remove spaces
        prefs.setdefault(user,{}) # make it a nested dicitonary
        prefs[user][movies[movieid]]=float(rating)
    
    #return a dictionary of preferences
    return prefs

def stats_helper(prefs):
    ''' Computes descriptive analytics:
        -- Average ratings
        -- Number of ratings
        -- Number of items
        -- Dictionary of items with thier ratings

        Parameters:
        -- prefs: dictionary containing user-item matrix
        
        
        Returns:
        -- List of computed values:
            avg_ratings, count_ratings, all_ratings,items_count, items_ratings

    '''
    avg_ratings = []
    count_ratings = 0 
    all_ratings = []
    items_count = {}
    items_ratings = {}

    
    for user in prefs:
        avg_ratings.append(np.mean(list(prefs[user].values())))
        for item in prefs[user]:
            count_ratings +=1
            all_ratings.append(prefs[user][item])
            items_count[item] = items_count.get(item, 0)+1
            items_ratings[item] = items_ratings.get(item,0) + prefs[user][item]
    
    return [avg_ratings, count_ratings, all_ratings,items_count, items_ratings]
    
def data_stats(prefs, filename):
    ''' Computes/prints descriptive analytics:
        -- Total number of users, items, ratings
        -- Overall average rating, standard dev (all users, all items)
        -- Average item rating, standard dev (all users)
        -- Average user rating, standard dev (all items)
        -- Matrix ratings sparsity
        -- Ratings distribution histogram (all users, all items)

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- filename: string containing name of file being analyzed
        
        Returns:
        -- None

    '''
    avg_ratings,count_ratings, all_ratings,items_count, items_ratings, \
    = stats_helper(prefs)
    
    all_avg_ratings =[]
    for item in items_count:
        all_avg_ratings.append(items_ratings[item]/items_count[item])
        
    #descriptive data 
    print ("Stats for:", filename)
    print("Number of users:", len(prefs))
    print("Number of items:", len(items_count))
    print("Number of ratings:", count_ratings)
    print("Overall average rating: %.2f out of 5, and std dev of %.2f" % 
          (np.mean(all_ratings), np.std(all_ratings)))
    print(("Average item rating: %.2f out of 5, and std dev of %.2f"% 
           (np.mean(all_avg_ratings), np.std(all_avg_ratings))))
    print(("Average user rating: %.2f out of 5, and std dev of %.2f"% 
           (np.mean(avg_ratings),np.std(avg_ratings))))
    print("User-Item Matrix Sparsity: %.2f%%"%
          (float(1-(count_ratings/(len(prefs)*len(items_count))))*100))
    print()
    
    #histogram
    hist,bins = np.histogram(all_ratings, bins=[1,2,3,4,5])
    xy = plt.gca()
    xy.set_xlim([1,5])
    xy.set_ylim(0,max(hist))
    xy.set_xticks([1,2,3,4,5])
    plt.hist(all_ratings,bins=[1,2,3,4,5], color="c")
    xy.set_facecolor('black')
    plt.title("Ratings Histogram")
    plt.xlabel("Rating")
    plt.ylabel("Number of user ratings")
    plt.grid()
    plt.show()
    
    
def popular_items(prefs, filename):
    ''' Computes/prints popular items analytics    
        -- popular items: most rated (sorted by # ratings)
        -- popular items: highest rated (sorted by avg rating)
        -- popular items: highest rated items that have at least a 
                          "threshold" number of ratings
        
        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- filename: string containing name of file being analyzed
        
        Returns:
        -- None

    '''
    avg_ratings,count_ratings, all_ratings,items_count, items_ratings, \
    = stats_helper(prefs)
    all_avg_ratings =[]
    item_count_list = []
    for item in items_count:
        all_avg_ratings.append((items_ratings[item]/items_count[item], item))
        item_count_list.append((items_count[item], item))
    
    item_count_list.sort()
    item_count_list.reverse()
    all_avg_ratings.sort()
    all_avg_ratings.reverse()
    
    num_popular_items = min(5, len(items_ratings))
    print("Popular items -- most rated: \nTitle \t\t\t\t\t #Ratings \t Avg Rating")
    
    for i in range(num_popular_items):
        print("%s \t %s \t\t %.2f"% (item_count_list[i][1].ljust(25), \
                                     str((item_count_list[i][0]))+' '*5, \
                                         float(items_ratings[item_count_list[i][1]]\
                                               /items_count[item_count_list[i][1]])))
    print()
    
    print("Popular items -- highest rated: \nTitle \t\t\t\t\t Avg Rating \t #Ratings")
    for i in range(num_popular_items):
        print("%s \t %.2f \t\t %s"%(all_avg_ratings[i][1].ljust(25),\
                                    float(all_avg_ratings[i][0]), \
                                        items_count[all_avg_ratings[i][1]]))
    print()
    

    overall_best_items = []
    for i in range(len(all_avg_ratings)):
        if items_count[all_avg_ratings[i][1]]>= num_popular_items:
            overall_best_items.append(all_avg_ratings[i])
   
    print("Overall best rated items (number of ratings>=%d): "
          "\nTitle \t\t\t\t\t Avg Rating \t #Ratings "%(num_popular_items))
    for i in range(num_popular_items):
        print("%s \t %.2f \t\t %s" % (overall_best_items[i][1].ljust(25), \
                                      overall_best_items[i][0], \
                                          items_count[overall_best_items[i][1]]))
        
    
def sim_distance(prefs,person1,person2,n=25): #default weight 
    '''
        Calculate Euclidean distance similarity 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person1: string containing name of user 1
        -- person2: string containing name of user 2
        -- n: significance weighting
        
        Returns:
        -- Euclidean distance similarity as a float
        
    '''
    
    # Get the list of shared_items
    si={}
    for item in prefs[person1]: 
        if item in prefs[person2]: 
            si[item]=1
    
    # if they have no ratings in common, return 0
    factor = 1
    if len(si)==0: 
        return 0
    if len(si) < n:
        factor = len(si)/n
    # Add up the squares of all the differences
    sum_of_squares=sum([pow(prefs[person1][item]-prefs[person2][item],2) 
                        for item in prefs[person1] if item in prefs[person2]])


        
    return factor *(1/(1+sqrt(sum_of_squares)))

def sim_pearson(prefs,p1,p2,n=1):
    '''
        Calculate Pearson Correlation similarity 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person1: string containing name of user 1
        -- person2: string containing name of user 2
        -- n: significance weighting
        
        Returns:
        -- Pearson Correlation similarity as a float
        
    '''
    

    si={p1:[], p2:[]}
    num = 0
    den1 = 0
    den2 = 0
    
    for item in prefs[p1]: 
        if item in prefs[p2]: 
            si[p1].append(prefs[p1][item])
            si[p2].append(prefs[p2][item])
    factor =1
    if (si[p1])==[]: 
        return 0
    if len(si[p1]) < n:
        factor = len(si[p1])/n

    p1_avg = np.mean(si[p1])
    p2_avg = np.mean(si[p2])
    
    for item in prefs[p1]:
        if item in prefs[p2]:
            num += (prefs[p1][item] - p1_avg)*(prefs[p2][item] - p2_avg)
            den1 += (prefs[p1][item] - p1_avg)**2
            den2 += (prefs[p2][item]- p2_avg)**2
            
    if (sqrt(den1*den2)==0):
        return 0
    else:
        return factor * (num/(sqrt(den1*den2)))



def getRecommendationsSim(prefs,itemMatch,user,thres=0.5): #default threshold
    '''
    Quicker calculation of recommendations for a given user 
    -- prefs: dictionary containing user-item matrix
    -- person: string containing name of user
    -- similarity: function to calc similarity (sim_pearson is default)
    -- thre: similarity threshold
    Returns:
        -- A list of recommended items with 0 or more tuples, 
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
    '''
  
    
    sim=itemMatch[user]
    
    totals={}
    simSums={}
    for other in prefs:
      # don't compare me to myself
        if other==user: 
            continue
        
      
        similarity = itemMatch[user]
        sim=0
        for (siml,user2) in similarity:
            if user2 == other:
                sim=siml
            else:
                continue
    
        if sim<=thres: continue
        for item in prefs[other]:
            
            # only score movies I haven't seen yet
            if item not in prefs[user] or prefs[user][item]==0:
                # Similarity * Score
                totals.setdefault(item,0)
                totals[item]+=prefs[other][item]*sim
                # Sum of similarities
                simSums.setdefault(item,0)
                simSums[item]+=sim
  
    # Create the normalized list
    rankings=[(total/simSums[item],item) for item,total in totals.items()]
  
    # Return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings
  


# def getRecommendations(prefs,person,similarity=sim_pearson,thres=0): #default threshold
#     '''
#         Calculates recommendations for a given user 

#         Parameters:
#         -- prefs: dictionary containing user-item matrix
#         -- person: string containing name of user
#         -- similarity: function to calc similarity (sim_pearson is default)
        
#         Returns:
#         -- A list of recommended items with 0 or more tuples, 
#            each tuple contains (predicted rating, item name).
#            List is sorted, high to low, by predicted rating.
#            An empty list is returned when no recommendations have been calc'd.
        
#     '''
    
#     totals={}
#     simSums={}
#     for other in prefs:
#       # don't compare me to myself
#         if other==person: 
#             continue
#         sim=similarity(prefs,person,other)
    
#         # ignore scores less than threshold
#         if sim<=thres: continue
#         for item in prefs[other]:
            
#             # only score movies I haven't seen yet
#             if item not in prefs[person] or prefs[person][item]==0:
#                 # Similarity * Score
#                 totals.setdefault(item,0)
#                 totals[item]+=prefs[other][item]*sim
#                 # Sum of similarities
#                 simSums.setdefault(item,0)
#                 simSums[item]+=sim
  
#     # Create the normalized list
#     rankings=[(total/simSums[item],item) for item,total in totals.items()]
  
#     # Return the sorted list
#     rankings.sort()
#     rankings.reverse()
#     return rankings

    
# def get_all_UU_recs(prefs, sim=sim_pearson, num_users=10, top_N=5):
#     ''' 
#     Print user-based CF recommendations for all users in dataset

#     Parameters
#     -- prefs: nested dictionary containing a U-I matrix
#     -- sim: similarity function to use (default = sim_pearson)
#     -- num_users: max number of users to print (default = 10)
#     -- top_N: max number of recommendations to print per user (default = 5)

#     Returns: None
#     '''
    
#     for user in prefs:
#         print('User-based CF recs for %s: ' % (user), 
#                        getRecommendations(prefs, user,similarity=sim)) 
#     print()
        
# def loo_cv(prefs, metric, sim, algo):
#     """
#     Leave_One_Out Evaluation: evaluates recommender system ACCURACY
     
#      Parameters:
#          prefs dataset: critics, ml-100K, etc.
#          metric: MSE, MAE, RMSE, etc.
#          sim: distance, pearson, etc.
#          algo: user-based recommender, item-based recommender, etc.
     
#     Returns:
#          error_total: MSE, MAE, RMSE totals for this set of conditions
#          error_list: list of actual-predicted differences
    
    
#     Algo Pseudocode ..
#     Create a temp copy of prefs
    
#     For each user in temp copy of prefs:
#       for each item in each user's profile:
#           delete this item
#           get recommendation (aka prediction) list
#           restore this item
#           if there is a recommendation for this item in the list returned
#               calc error, save into error list
#           otherwise, continue
      
#     return mean error, error list
#     """
#     if algo == getRecommendations :
        
#         error_list = []
#         temp_prefs = prefs.copy()
#         for user in temp_prefs:
#             for item in list(temp_prefs[user].keys()):
#                 item_del = prefs[user][item]
#                 del temp_prefs[user][item]
#                 recc_list = getRecommendations(temp_prefs, user,similarity=sim)
                   
                    
#                 temp_prefs[user][item] = item_del

                
#                 for recc in recc_list:
#                     if item in recc:
#                         if item == recc[1]:
#                             err = (recc[0]-item_del)**2
#                             print('User: %s, Item: %s, Prediction: %.5f, Actual: %.5f, Error: %.5f'% (user, item, \
#                                       recc[0], item_del, err))
#                             error_list.append(err)
            
                   
#         error_total = np.mean(error_list)
        
#         if metric == "MSE":
#             return error_total , error_list
               

def topMatches(prefs,person,similarity=sim_pearson, n=5):
    '''
        Returns the best matches for person from the prefs dictionary

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)
        -- n: number of matches to find/return (5 is default)
        
        Returns:
        -- A list of similar matches with 0 or more tuples, 
           each tuple contains (similarity, item name).
           List is sorted, high to low, by similarity.
           An empty list is returned when no matches have been calc'd.
        
    '''     
    scores=[(similarity(prefs,person,other),other) 
                    for other in prefs if other!=person]
    scores.sort()
    scores.reverse()
    return scores[0:n]

def transformPrefs(prefs):
    '''
        Transposes U-I matrix (prefs dictionary) 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        
        Returns:
        -- A transposed U-I matrix, i.e., if prefs was a U-I matrix, 
           this function returns an I-U matrix
        
    '''     
    result={}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item,{})
            # Flip item and person
            result[item][person]=prefs[person][item]
    return result

def calculateSimilarItems(prefs,n=100,similarity=sim_pearson):

    '''
        Creates a dictionary of items showing which other items they are most 
        similar to. 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- n: number of similar matches for topMatches() to return
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A dictionary with a similarity matrix
        
    '''     
    result={}
    # Invert the preference matrix to be item-centric
    itemPrefs=transformPrefs(prefs)
    c=0
    for item in itemPrefs:
      # Status updates for larger datasets
        c+=1
        if c%100==0: 
            print ("%d / %d" % (c,len(itemPrefs)))
            
        # Find the most similar items to this one
        scores=topMatches(itemPrefs,item,similarity,n=n)
        result[item]=scores
    return result
def calculateSimilarUsers(prefs,n=100,similarity=sim_pearson):

    '''
        Creates a dictionary of items showing which other users they are most 
        similar to. 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- n: number of similar matches for topMatches() to return
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A dictionary with a similarity matrix
        
    '''     
    result={}
    # Invert the preference matrix to be item-centric
    
    c=0
    for user in prefs:
      # Status updates for larger datasets
        c+=1
        if c%100==0: 
            print ("%d / %d" % (c,len(prefs)))
            
        # Find the most similar items to this one
        scores=topMatches(prefs,user,similarity,n=n)
        result[user]=scores
    
    return result

def getRecommendedItems(prefs,itemMatch,user,thres=0.5): #default threshold is zero
    '''
        Calculates recommendations for a given user 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A list of recommended items with 0 or more tuples, 
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
        
    '''
    userRatings=prefs[user]
    scores={}
    totalSim={}
    # Loop over items rated by this user
    for (item,rating) in userRatings.items( ):
  
      # Loop over items similar to this one
        for (similarity,item2) in itemMatch[item]:
    
            # Ignore if this user has already rated this item
            if item2 in userRatings: continue
            # ignore scores of zero or lower
            if similarity<=thres: continue            
            # Weighted sum of rating times similarity
            scores.setdefault(item2,0)
            scores[item2]+=similarity*rating
            # Sum of all the similarities
            totalSim.setdefault(item2,0)
            totalSim[item2]+=similarity
  
    # Divide each total score by total weighting to get an average

    rankings=[(score/totalSim[item],item) for item,score in scores.items( )]    
  
    # Return the rankings from highest to lowest
    rankings.sort( )
    rankings.reverse( )
    return rankings
           
def get_all_II_recs(prefs, itemsim, sim_method, num_users=10, top_N=5):
    ''' 
    Print item-based CF recommendations for all users in dataset

    Parameters
    -- prefs: U-I matrix (nested dictionary)
    -- itemsim: item-item similarity matrix (nested dictionary)
    -- sim_method: name of similarity method used to calc sim matrix (string)
    -- num_users: max number of users to print (integer, default = 10)
    -- top_N: max number of recommendations to print per user (integer, default = 5)

    Returns: None
    
    '''
    for user in prefs:
        print ('Item-based CF recs for %s, %s: ' % (user, sim_method), 
                       getRecommendedItems(prefs, itemsim, user)) 
    print()
    
def loo_cv_sim(prefs, sim, algo, sim_matrix):
    """
    Leave-One_Out Evaluation: evaluates recommender system ACCURACY
     
     Parameters:
         prefs dataset: critics, etc.
	 
	 sim: distance, pearson, etc.
	 algo: user-based recommender, item-based recommender, etc.
     sim_matrix: pre-computed similarity matrix
	 
    Returns:
         error_total: MSE, or MAE, or RMSE totals for this set of conditions
	 error_list: list of actual-predicted differences
    """
   
    error_list = []
    error_listabs = []
    temp_prefs = prefs.copy()

        
    for user in temp_prefs:
        for item in list(temp_prefs[user].keys()):
            item_del = prefs[user][item]
            del temp_prefs[user][item]
            
            if algo == getRecommendedItems:
                recc_list = getRecommendedItems(temp_prefs,sim_matrix,user)
            if algo == getRecommendationsSim:
                recc_list = getRecommendationsSim(temp_prefs,sim,user)   

            
            for recc in recc_list:
                if item in recc:
                    if item == recc[1]:
                        err = (recc[0]-item_del)**2
                        err_1 = abs(recc[0]-item_del)
                        print('User: %s, Item: %s, Prediction: %.5f, Actual: %.5f, Error: %.5f'% (user, item, \
                                    recc[0], item_del, err))
                        error_listabs.append(err_1)
                        error_list.append(err)

    
           
    error_total = np.mean(error_list)
    error_totalabs = np.mean(error_listabs)
    MSE = error_total
    MSE_list = error_list
    MAE = error_totalabs
    MAE_list = error_listabs
    RMSE = sqrt(error_total)
    RMSE_list = error_list
    return MSE, MSE_list, MAE, MAE_list, RMSE, RMSE_list
        
# def print_loocv_results(sq_diffs_info):
#     ''' Print LOOCV SIM results '''

#     error_list = []
#     for user in sq_diffs_info:
#         for item in sq_diffs_info[user]:
#             for data_point in sq_diffs_info[user][item]:
#                 #print ('User: %s, Item: %s, Prediction: %.5f, Actual: %.5f, Error: %.5f' %\
#             #      (user, item, data_point[0], data_point[1], data_point[2]))                
#                 error_list.append(data_point[2]) # save for MSE calc
                
#     #print()
#     error = sum(error_list)/len(error_list)          
#     #print ('MSE =', error)
    
#     return(error, error_list)

def main():
    ''' User interface for Python console '''
    
    # Load ml-100k dict from file
    path = os.getcwd() # this gets the current working directory
                       # you can customize path for your own computer here
    print('\npath: %s' % path) # debug
    done = False
    prefs = {}
    
    while not done: 
        print()
        # Start a simple dialog
        file_io = input('RML(ead ml100K data)?, \n'
                        'P(rint) the U-I matrix?, \n'
                        'V(alidate) the dictionary?, \n'
                        'S(tats) print?, \n'
                        'D(istance) ml-100k data?, \n'
                        'PC(earson Correlation) ml-100k data?, \n'
                        # 'SC(earman Correlation) critics data?, \n'
                        # 'JC(ccard Correlation) critics data?, \n'
                        # 'CC(osine Correlation) critics data?, \n'
                        # 'U(ser-based CF Recommendations)?, \n'
                        # 'I(tem-based CF Recommendations)?, \n'
                        # 'LCV(eave one out cross-validation)?, \n'
                        'SIMU(ilarity matrix user) calc for User-based recommend?\n'
                        'Sim(ilarity matrix) calc for Item-based recommender? \n'
                       
                        'LCVSIM(eave one out cross-validation)? \n'
                        ' ==>> ')
        
        if file_io == 'RML' or file_io == 'rml':
            print()
            file_dir = 'data/ml-100k/' # path from current directory
            datafile = 'u.data'
            itemfile = 'u.item'
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)
            
            print('Number of users: %d\nList of users [0:10]:' 
                      % len(prefs), list(prefs.keys())[0:10] )
            
            
        elif file_io == 'P' or file_io == 'p':
            # print the u-i matrix
            print()
            if len(prefs) > 0:
                print ('Printing "%s" dictionary from file' % datafile)
                print ('User-item matrix contents: user, item, rating')
                for user in prefs:
                    for item in prefs[user]:
                        print(user, item, prefs[user][item])
            else:
                print ('Empty dictionary, RML(ead ml100K) in some data!')
                
        elif file_io == 'V' or file_io == 'v':      
            print()
            if len(prefs) > 0:
                # Validate the dictionary contents ..
                print ('Validating "%s" dictionary from file' % datafile)
                print ("ml-100k['196']['Kolya (1996)'] =", 
                       prefs['196']['Kolya (1996)']) # ==> 3.0
                print ("ml-100k['197']['Terminator 2: Judgment Day (1991)'] =", 
                       prefs['197']['Terminator 2: Judgment Day (1991)']) # ==> 5.0
            else:
                 print ('Empty dictionary, RML(ead ml100K) in some data!')
                
        elif file_io == 'S' or file_io == 's':
            print()
            filename = 'ml-100k.data'
            if len(prefs) > 0:
                data_stats(prefs, filename)
                popular_items(prefs, filename)
            else: # Make sure there is data  to process ..
                 print ('Empty dictionary, RML(ead ml100K) in some data!')
                 
        elif file_io == 'D' or file_io == 'd':
            print()
            if len(prefs) > 0:            
                print()
                print('User-User distance similarities:')
                # Calc distance for all users
                for user in list(prefs.keys()):
                    for other in list(prefs.keys()):
                        if user==other:
                            continue
                        print('Distance sim for %s & %s: %s'
                             %(user, other, sim_distance(prefs, user,other)))
                
                print()
            else: # Make sure there is data  to process ..
                 print ('Empty dictionary, RML(ead ml100K) in some data!')
                 
        elif file_io == 'PC' or file_io == 'pc':
            print()
            if len(prefs) > 0:
                print('Pearson for all users:')
                # Calc Pearson for all users
                for user in list(prefs.keys()):
                    for other in list(prefs.keys()):
                        if user==other:
                            continue
                        print('Pearson sim for %s & %s: %s'%(user, other, sim_pearson(prefs, user,other)))
                
                print()
            else: # Make sure there is data  to process ..
                print ('Empty dictionary, RML(ead ml100K) in some data!')
       
        # elif file_io == 'U' or file_io == 'u':
        #     print()
        #     if len(prefs) > 0:             
        #         print ('Example:')
        #         user_name = 'Toby'
        #         print ('User-based CF recs for %s, sim_pearson: ' % (user_name), 
        #                 getRecommendations(prefs, user_name)) 
                       
        #         print ('User-based CF recs for %s, sim_distance: ' % (user_name),
        #                 getRecommendations(prefs, user_name, similarity=sim_distance)) 
        #         print()
                
        #         print('User-based CF recommendations for all users:')
        #         # Calc User-based CF recommendations for all users
        #         print('Using sim_pearson:')
        #         get_all_UU_recs(prefs)
        #         print('Using sim_distance:')
        #         get_all_UU_recs(prefs,sim=sim_distance)
             
        #         print()
        # elif file_io == 'LCV' or file_io == 'lcv':
        #     print()
        #     if len(prefs) > 0:             
        
        #         print('LOO_CV Evaluation for User-based')
                
        #         sim = sim_pearson
        #         algo = getRecommendations
        #         MSE, MSE_list = loo_cv(prefs,'MSE', sim, algo)
        #         print('MSE for critics: %.10f, len(SE list): %d, using sim_pearson'%(MSE, len(MSE_list)))
        #         print()
        #         sim = sim_distance
        #         MSE, MSE_list = loo_cv(prefs,'MSE', sim, algo)
        #         print('MSE for critics: %.10f, len(SE list): %d, using sim_distance'%(MSE,len(MSE_list)))
                
                

        elif file_io == 'SIMU' or file_io == 'simu':
            print()
            if len(prefs) > 0: 
                ready = False # sub command in progress
                sub_cmd = input('RD(ead) distance or RP(ead) pearson or WD(rite) distance or WP(rite) pearson?')
                try:
                    if sub_cmd == 'RD' or sub_cmd == 'rd':
                        # Load the dictionary back from the pickle file.
                        usersim = pickle.load(open( "save_usersim_distance.p", "rb" ))
                        sim_method = 'sim_distance'
    
                    elif sub_cmd == 'RP' or sub_cmd == 'rp':
                        # Load the dictionary back from the pickle file.
                        usersim = pickle.load(open( "save_usersim_pearson.p", "rb" ))  
                        sim_method = 'sim_pearson'
                    

                    elif sub_cmd == 'WD' or sub_cmd == 'wd':
                        
                        usersim = calculateSimilarUsers(prefs,similarity=sim_distance)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(usersim, open( "save_usersim_distance.p", "wb" ))
                        sim_method = 'sim_distance'
                        
                    elif sub_cmd == 'WP' or sub_cmd == 'wp':
                        
                        usersim = calculateSimilarUsers(prefs,similarity=sim_pearson)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(usersim, open( "save_usersim_pearson.p", "wb" )) 
                        sim_method = 'sim_pearson'
                    
                  
                    else:
                        print("Sim sub-command %s is invalid, try again" % sub_cmd)
                        continue
                    
                    ready = True # sub command completed successfully
                    
                except Exception as ex:
                    print ('Error!!', ex, '\nNeed to W(rite) a file before you can R(ead) it!'
                           ' Enter Sim(ilarity matrix) again and choose a Write command')
                    print()
                if len(usersim) > 0 and ready == True: 
                # Only want to print if sub command completed successfully
                    print('Similarity matrix based on %s, len = %d' 
                        % (sim_method, len(usersim)))
                    print()
                    ##
                    ## enter new code here, or call a new function, 
                    if (sim_method=='sim_distance'):
                        result = calculateSimilarUsers(prefs, n=100, similarity=sim_distance)
                 
                    else:
                        result = calculateSimilarUsers(prefs)
                    for user in result:
                        print(user, result[user])
                    ##    to print the sim matrix
                    ##
                print()
            else:
                print ('Empty dictionary, RML(ead ml100K) in some data!')
        elif file_io == 'Sim' or file_io == 'sim':
            print()
            if len(prefs) > 0: 
                ready = False # sub command in progress
                sub_cmd = input('RD(ead) distance or RP(ead) pearson or WD(rite) distance or WP(rite) pearson?')
                try:
                    if sub_cmd == 'RD' or sub_cmd == 'rd':
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(open( "save_itemsim_distance.p", "rb" ))
                        sim_method = 'sim_distance'
    
                    elif sub_cmd == 'RP' or sub_cmd == 'rp':
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(open( "save_itemsim_pearson.p", "rb" ))  
                        sim_method = 'sim_pearson'
                    
                 

                    elif sub_cmd == 'WD' or sub_cmd == 'wd':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(prefs,similarity=sim_distance)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(itemsim, open( "save_itemsim_distance.p", "wb" ))
                        sim_method = 'sim_distance'
                        
                    elif sub_cmd == 'WP' or sub_cmd == 'wp':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(prefs,similarity=sim_pearson)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(itemsim, open( "save_itemsim_pearson.p", "wb" )) 
                        sim_method = 'sim_pearson'
               
                    
                    else:
                        print("Sim sub-command %s is invalid, try again" % sub_cmd)
                        continue
                    
                    ready = True # sub command completed successfully
                    
                except Exception as ex:
                    print ('Error!!', ex, '\nNeed to W(rite) a file before you can R(ead) it!'
                           ' Enter Sim(ilarity matrix) again and choose a Write command')
                    print()
                

                if len(itemsim) > 0 and ready == True: 
                    # Only want to print if sub command completed successfully
                    print ('Similarity matrix based on %s, len = %d' 
                           % (sim_method, len(itemsim)))
                    print()
                    ##
                    ## enter new code here, or call a new function, 
                    if (sim_method=='sim_distance'):
                        result = calculateSimilarItems(prefs, n=100, similarity=sim_distance)
                  
                    else:
                        result = calculateSimilarItems(prefs)
                    for item in result:
                        print(item, result[item])
                    ##    to print the sim matrix
                    ##
                print()
            else:
                print ('Empty dictionary, RML(ead ml100K) in some data!')  
                
        elif file_io == 'I' or file_io == 'i':
            print()
            
            if len(prefs) > 0 and len(itemsim) > 0:                
                print ('Example:')
                user_name = 'Toby'
                
    
                print ('Item-based CF recs for %s, %s: ' % (user_name, sim_method), 
                       getRecommendedItems(prefs, itemsim, user_name)) 
                
    
                print()
                
                print('Item-based CF recommendations for all users:')
               
                if  sim_method == "sim_distance":
               
                    get_all_II_recs(prefs, itemsim, sim_method) # num_users=10, and top_N=5 by default  '''
               
                
                if  sim_method == "sim_pearson":
                
                    get_all_II_recs(prefs, itemsim, sim_method)
                
                    
                print()
                
            else:
                if len(prefs) == 0:
                    print ('Empty dictionary, RML(ead ml100K) in some data!')
                else:
                    print ('Empty similarity matrix, use SIMU(ilarity) to create a sim matrix!') 
        elif file_io == 'LCVSIM' or file_io == 'lcvsim':
            print()
            file_io = input('Enter U(ser) or I(tem) algo:')
            if file_io == 'U' or file_io == 'u':
                if len(prefs) > 0 and usersim !={}:             
                    print('LOO_CV_SIM Evaluation')
                   
                    prefs_name = 'ML-100K'
                   
                    algo = getRecommendationsSim #user-based

                    if sim_method == 'sim_pearson': 
                        sim = sim_pearson
                        MSE, MSE_list, MAE, MAE_list, RMSE,RMSE_list = loo_cv_sim(prefs, sim, algo, usersim)
                        print('MSE for %s: %.5f, len(MSE list): %d, MAE: %.5f, len(MAE list): %d,\
                            RMSE: %.5f, len(RMSE list): %d, using %s' %(prefs_name, MSE, len(MSE_list),MAE,len(MAE_list),RMSE, len(RMSE_list), sim))
                        print()
                    
                    if sim_method == 'sim_distance':
                        sim = sim_distance
                        MSE,MSE_list, MAE, MAE_list, RMSE,RMSE_list = loo_cv_sim(prefs, sim, algo, usersim)
                        print('MSE for %s:%.5f, len(MSE list): %d, MAE: %.5f, len(MAE list): %d,\
                            RMSE: %.5f, len(RMSE list): %d, using %s' %(prefs_name,MSE, len(MSE_list),MAE,len(MAE_list),RMSE, len(RMSE_list), sim))
                 
                    else:
                        print('Run Sim(ilarity matrix) command to create/load Sim matrix!')
                
                else:
                    print ('Empty dictionary, run RML(ead ml100K) OR Empty Sim Matrix, run Sim!')
            elif file_io == 'I' or file_io == 'i':    
                if len(prefs) > 0 and itemsim !={}:             
                    print('LOO_CV_SIM Evaluation')
                   
                    prefs_name = 'ML-100K'

                    algo = getRecommendedItems ## Item-based recommendation
                    
                    
                    if sim_method == 'sim_pearson': 
                        sim = sim_pearson
                        MSE, MSE_list, MAE, MAE_list, RMSE,RMSE_list = loo_cv_sim(prefs, sim, algo, itemsim)
                        print('MSE for %s: %.5f, len(MSE list): %d, MAE: %.5f, len(MAE list): %d,\
                            RMSE: %.5f, len(RMSE list): %d, using %s' %(prefs_name, MSE, len(MSE_list),MAE,len(MAE_list),RMSE, len(RMSE_list), sim))
                        print()
                    
                    if sim_method == 'sim_distance':
                        sim = sim_distance
                        MSE, MSE_list, MAE, MAE_list, RMSE,RMSE_list = loo_cv_sim(prefs, sim, algo, itemsim)
                        print('MSE for %s: %.5f, len(MSE list): %d, MAE: %.5f, len(MAE list): %d,\
                            RMSE: %.5f, len(RMSE list): %d, using %s' %(prefs_name, MSE, len(MSE_list),MAE,len(MAE_list),RMSE, len(RMSE_list), sim))
                        print()
               
                    else:
                        print('Run Sim(ilarity matrix) command to create/load Sim matrix!')
                 
                else:
                    print ('Empty dictionary, run RML(ead ml100K) OR Empty Sim Matrix, run Sim!')
            
        else:
            done = True
    
    print('\nGoodbye!')
        
if __name__ == '__main__':
    main()