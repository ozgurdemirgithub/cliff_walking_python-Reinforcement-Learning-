#CLIFF-WALKING
import numpy as np, matplotlib.pyplot as plt #importing necessary packages
rows,cols,direction = 4,12,4 # set the values for qtable and possible movements

episodes, epsilon, alpha, gamma = 500, .1, .5, .99
#episodes: num of trainings
#epsilon: explore/exploit value
#alpha: determines convergence ratio 
#gamma: determines significance of further rewards
       
def get_location(position, q_table):
    (i , j) = position #coordinate value of agent/used for determining movement
    location = int(12 * i + j) #location in the cliff/used for determining reward/punishment
    location_movement = q_table[0:48, location]
    max_location = np.amax(location_movement) 
    return location, max_location

def get_reward(location): 
    end = True if  48 >= location >= 37 else False #game ends at cliff and final locations
    reward = -100 if( 46 >= location >= 37) else -1 # -100 reward for falling to cliff otherwise -1 
    return reward, end

def epsilon_greedy(location, gridworld, epsilon):
    random  = np.random.random() #[0,1.0)
    movement = np.random.randint(direction) if random < epsilon else np.argmax(gridworld[0:48,location]) 
    return movement

def movement_(position, movement):
    (i,j) = position #coordinate value of agent/used for determining movement
    south = [movement == 0, i < 3 ] #down
    east  = [movement == 1, j < 11] #right
    west  = [movement == 2, j > 0 ] #left
    north = [movement == 3, i > 0 ] #up
    if all(north):
        i -= 1
    if all(west):
        j -= 1  
    if all(east):
        j += 1
    if all(south):
        i += 1
    position = (i,j)
    return position

def bellman_update_gridworld(gridworld, location, movement, reward, new_location, gamma, alpha):#bellman update
    #Q[y,x,a] = Q[y,x,a] + alpha*(reward + gamma*Qs1a1 - Q[y,x,a]) 
    updated_gridworld = gridworld[movement, location] + alpha * (reward + (gamma * new_location) - gridworld[movement, location])
    gridworld[movement, location] = updated_gridworld
    return gridworld    

def cliff(position, cliff): #used for visualizing the path that alghoritigms took.
    (i, j) = position #coordinate value of agent/used for determining movement
    cliff[i][j] = 1 # mark the location with 1 for differentiating the path taken.
    return cliff #visual environment

def sarsa(episodes, gamma, alpha, epsilon):#on-policy
    gridworld = np.zeros((4, cols * rows)) #creating a q_table with zeros
    reward_sarsa = [] # list for appending reward data
    for episode in range(episodes): #start training
        position, game_end, reward_sum = (3, 0), False, 0
        path = cliff(position, np.zeros((rows,cols)))
        while(game_end == False):
            location, _ = get_location(position, gridworld)
            movement   = epsilon_greedy(location, gridworld,epsilon)
            position = movement_(position, movement)
            path = cliff(position, path)
            new_location, _  = get_location(position, gridworld)
            reward, game_end = get_reward(new_location)
            reward_sum += reward 
            next_movement = epsilon_greedy(new_location, gridworld,epsilon)
            new_location = gridworld[next_movement][new_location] 
            gridworld = bellman_update_gridworld(gridworld, location, movement, reward, new_location, gamma, alpha)
            location,movement = new_location,next_movement
        print("SARSA-Safe Path" ,"\n" ,path) if episode == 499 else None #visualizing the path that alghoritigms took  
        reward_sarsa.append(reward_sum) #appending reward data
    return gridworld, reward_sarsa

def qlearning(episodes, gamma, alpha, epsilon):#off-policy
    q_table = np.zeros((4, cols * rows)) #creating a q_table with zeros
    reward_qlearning = [] #list for appending reward data
    for episode in range(episodes): #start training
        position, game_end, reward_sum  = (3, 0), False, 0, 
        path = cliff(position, np.zeros((rows,cols)))
        while(game_end == False):
            location, _ = get_location(position, q_table)
            movement   = epsilon_greedy(location, q_table,epsilon)
            position = movement_(position, movement)
            path = cliff(position , path)
            new_location, max_location = get_location(position, q_table)
            reward, game_end = get_reward(new_location)
            reward_sum += reward 
            gridworld = bellman_update_gridworld(q_table, location, movement, reward, max_location, gamma, alpha)
            location   = new_location
        print("Qlearning-Optimal Path" ,"\n" , path) if episode == 499 else None #visualizing the path that alghorithms took 
        reward_qlearning.append(reward_sum) 
    return gridworld, reward_qlearning

def plot_cliff_walking(reward_qlearning, reward_sarsa):#plotting results
    rewards_qlearning, rewards_sarsa,count  = [],[],0

    for reward in reward_qlearning:
        count +=1
        if reward >= -138:   #normalizing results 
            if count%20 == 0:#normalizing results
                rewards_qlearning.append(reward)
    for reward in reward_sarsa:
        count +=1
        if reward >= -138:   #normalizing results   
            if count%20 == 0:#normalizing results
                rewards_sarsa.append(reward)  
        
    plt.style.use ("seaborn-whitegrid")
    plt.title     ("Cliff-walking")
    plt.plot      (rewards_sarsa    , color="r" ,label = "SARSA" )
    plt.plot      (rewards_qlearning, color="k" ,label = "q_learning")
    plt.ylabel    ("Reward" + "\n" + "per" + "\n" + "episode", rotation = "horizontal")
    plt.xlabel    ("Episodes")
    plt.legend    (loc="best")
    
gridworld_sarsa    , reward_sarsa     = sarsa    (episodes, gamma, alpha,epsilon)
gridworld_qlearning, reward_qlearning = qlearning(episodes, gamma, alpha,epsilon)
plot_cliff_walking(reward_qlearning,reward_sarsa)