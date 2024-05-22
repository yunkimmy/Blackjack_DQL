from environment import blackjack
from agent import player

def print_hand(state):
    print("your hand: ", state["player"]["hand"])
    print("dealer hand: ", state["dealer"]["hand"])

done = False

env = blackjack()

state = env.reset()

while not state["turn"] and not done:
    print_hand(state)
    sum_hand = env.get_sum(state["player"])
    action = int(input("hit or stand? hit = 0, stand = 1: "))
    next_state, reward , done= env.player_interaction(state, action)
    state = next_state

if done:
    dealer_sum = env.get_sum(state["dealer"])
    player_sum = env.get_sum(state["player"])
    if env.is_bust(state["player"]):
        print("you bust, reward: {}".format(reward))
    elif env.is_bust(state["dealer"]):
        print("dealer bust, reward: {}".format(reward))
    elif dealer_sum > player_sum:
        print("dealer won, reward: {}".format(reward))
    elif player_sum > dealer_sum:
        print("player won, reward: {}".format(reward))
    else:
        print("draw, reward: {}".format(reward))

    print_hand(state)
        
    
    

if not done:
    new_state, reward, done = env.dealer_interaction(state)
    state = new_state

    dealer_sum = env.get_sum(state["dealer"])
    player_sum = env.get_sum(state["player"])

    if env.is_bust(state["dealer"]):
        print("dealer bust, reward: {}".format(reward))
    elif dealer_sum > player_sum:
        print("dealer won, reward: {}".format(reward))
    elif player_sum > dealer_sum:
        print("player won, reward: {}".format(reward))
    else:
        print("draw, reward: {}".format(reward))

    print_hand(state)

